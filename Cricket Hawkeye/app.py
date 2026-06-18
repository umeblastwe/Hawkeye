import os
import uuid
import cv2
import numpy as np
import time

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# ==========================================
# KALMAN FILTER FOR SMOOTHING
# ==========================================
class SimpleKalman:
    def __init__(self):
        self.q = 0.05
        self.r = 1.0
        self.x = 0.0
        self.p = 1.0
        self.k = 0.0

    def update(self, measurement):
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return int(self.x)


# ==========================================
# STRICT BALL CANDIDATE FILTER
# Players, fielders and umpires also move — so background subtraction
# alone is not enough. We add three extra filters here:
#   1. Circularity   — a real ball silhouette is near-perfectly round.
#                       A hand, leg or torso edge is not.
#   2. Size ceiling   — much tighter than before; a person-sized blob
#                       (arms moving, body shifting) is far bigger than
#                       a cricket ball even after camera distance.
#   3. Solidity       — contour area vs convex hull area; round objects
#                       are highly solid, limbs/edges are not.
# ==========================================
def is_ball_shaped(cnt):
    area = cv2.contourArea(cnt)
    if area < 4 or area > 140:          # tighter ceiling than before
        return False, 0

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False, 0

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.55:              # 1.0 = perfect circle
        return False, 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False, 0
    solidity = area / hull_area
    if solidity < 0.75:
        return False, 0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / max(h, 1)
    if aspect < 0.6 or aspect > 1.7:    # near-square bounding box
        return False, 0

    # Composite score — higher is more "ball-like"; used to pick the
    # best candidate when several pass the hard filters in one frame.
    score = circularity * solidity
    return True, score


# ==========================================
# TRAJECTORY-CONSISTENT BALL TRACKING
# Even after shape filtering, we may get a few false positives per
# video. This step builds the path by preferring candidates that
# continue the existing motion direction (a ball moves smoothly;
# random objects don't line up frame-to-frame).
# ==========================================
def detect_ball_opencv(video_path, max_seconds_budget=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(">>> ERROR: could not open video")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> Video: {total_frames} frames @ {fps:.1f}fps, {width}x{height}")

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=25, varThreshold=22, detectShadows=False
    )

    # Pitch-strip ROI — narrower than before. The ball during the
    # critical pitch->pad phase stays close to the pitch strip itself,
    # not out near the fielders/boundary.
    y1, y2 = int(height * 0.20), int(height * 0.85)
    x1, x2 = int(width * 0.30), int(width * 0.70)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    all_candidates = []   # list of lists: candidates[frame] = [(x,y,score), ...]
    frame_number = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if time.time() - start_time > max_seconds_budget:
            print(f">>> Time budget exceeded at frame {frame_number}")
            break

        roi = frame[y1:y2, x1:x2]
        fg_mask = subtractor.apply(roi)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_candidates = []
        for cnt in contours:
            ok, score = is_ball_shaped(cnt)
            if not ok:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00']) + x1
            cy = int(M['m01'] / M['m00']) + y1
            frame_candidates.append((cx, cy, score))

        all_candidates.append(frame_candidates)

    cap.release()
    print(f">>> Scanned {frame_number} frames, "
          f"frames with >=1 candidate: {sum(1 for c in all_candidates if c)}")

    # ── Build the smoothest possible path through candidates ──
    # Greedy nearest-motion-consistent selection: start from the frame
    # with the single best (highest-score) candidate, then walk forward
    # and backward picking whichever candidate best continues the
    # current velocity vector. This rejects one-off false detections
    # (a hand, a shadow) that don't line up with the ball's actual path.
    path = _build_consistent_path(all_candidates)

    print(f">>> Final consistent ball path length: {len(path)}")

    if len(path) < 3:
        return path

    kf_x = SimpleKalman()
    kf_y = SimpleKalman()
    kf_x.x = path[0]["x"]
    kf_y.x = path[0]["y"]

    smoothed = []
    for p in path:
        smoothed.append({
            "frame": p["frame"],
            "x": kf_x.update(p["x"]),
            "y": kf_y.update(p["y"])
        })

    return smoothed


def _build_consistent_path(all_candidates):
    """
    all_candidates[i] = list of (x, y, score) tuples detected in frame i+1
    Returns a clean list of {"frame": n, "x": .., "y": ..} dicts representing
    a single smooth trajectory, rejecting outlier blobs (players, shadows).
    """
    # Find the best seed frame: the frame with the highest-scoring
    # candidate, used as our starting anchor point.
    best_seed = None
    best_seed_idx = None
    for i, cands in enumerate(all_candidates):
        for c in cands:
            if best_seed is None or c[2] > best_seed[2]:
                best_seed = c
                best_seed_idx = i

    if best_seed is None:
        return []

    path = {best_seed_idx: (best_seed[0], best_seed[1])}

    MAX_JUMP = 60  # max pixels a ball can move between consecutive sampled frames

    # Walk forward from seed
    last_pos = (best_seed[0], best_seed[1])
    last_vel = (0, 0)
    for i in range(best_seed_idx + 1, len(all_candidates)):
        cands = all_candidates[i]
        if not cands:
            continue
        predicted = (last_pos[0] + last_vel[0], last_pos[1] + last_vel[1])
        best_c = None
        best_dist = None
        for c in cands:
            dist = ((c[0] - predicted[0]) ** 2 + (c[1] - predicted[1]) ** 2) ** 0.5
            if dist > MAX_JUMP:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_c = c
        if best_c is None:
            continue
        new_pos = (best_c[0], best_c[1])
        last_vel = (new_pos[0] - last_pos[0], new_pos[1] - last_pos[1])
        last_pos = new_pos
        path[i] = new_pos

    # Walk backward from seed
    last_pos = (best_seed[0], best_seed[1])
    last_vel = (0, 0)
    for i in range(best_seed_idx - 1, -1, -1):
        cands = all_candidates[i]
        if not cands:
            continue
        predicted = (last_pos[0] - last_vel[0], last_pos[1] - last_vel[1])
        best_c = None
        best_dist = None
        for c in cands:
            dist = ((c[0] - predicted[0]) ** 2 + (c[1] - predicted[1]) ** 2) ** 0.5
            if dist > MAX_JUMP:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_c = c
        if best_c is None:
            continue
        new_pos = (best_c[0], best_c[1])
        last_vel = (last_pos[0] - new_pos[0], last_pos[1] - new_pos[1])
        last_pos = new_pos
        path[i] = new_pos

    sorted_frames = sorted(path.keys())
    return [{"frame": f + 1, "x": path[f][0], "y": path[f][1]} for f in sorted_frames]


# ==========================================
# HAWK-EYE QUADRATIC TRAJECTORY PREDICTION
# ==========================================
def calculate_quadratic_path(points, width, height, batsman_hand="Right"):
    if len(points) < 4:
        return {
            "pitching": "UNKNOWN", "impact": "UNKNOWN", "wicket": "UNKNOWN", "decision": "NOT OUT"
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    dy = np.diff(ys)
    bounce_index = int(np.argmax(ys))
    for i in range(1, len(dy)):
        if dy[i-1] > 0 and dy[i] < 0:
            bounce_index = i
            break

    pitch_x = float(xs[bounce_index])
    pitch_y = float(ys[bounce_index])

    impact_index = min(bounce_index + 1, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    coeff = np.polyfit(ys, xs, 2)
    stump_y = height * 0.60
    projected_x = float(np.polyval(coeff, stump_y))

    stump_left = width * 0.465
    stump_right = width * 0.535
    uc_margin = width * 0.015

    leg_boundary = stump_left if batsman_hand == "Right" else stump_right
    pitched_outside_leg = (pitch_x < leg_boundary) if batsman_hand == "Right" else (pitch_x > leg_boundary)

    if pitched_outside_leg:
        pitching = "OUTSIDE LEG"
    elif stump_left <= pitch_x <= stump_right:
        pitching = "IN LINE"
    else:
        pitching = "OUTSIDE OFF"

    impact_in_line = stump_left - uc_margin <= impact_x <= stump_right + uc_margin
    impact = "IN LINE" if impact_in_line else ("OUTSIDE OFF" if impact_x > stump_right else "OUTSIDE LEG")

    if stump_left + uc_margin <= projected_x <= stump_right - uc_margin:
        wicket = "HITTING"
    elif (stump_left - uc_margin <= projected_x < stump_left + uc_margin) or \
         (stump_right - uc_margin < projected_x <= stump_right + uc_margin):
        wicket = "UMPIRE'S CALL"
    else:
        wicket = "MISSING"

    if pitched_outside_leg:
        decision = "NOT OUT"
    elif impact_x > stump_right + uc_margin or impact_x < stump_left - uc_margin:
        decision = "NOT OUT"
    elif wicket == "HITTING":
        decision = "OUT"
    elif wicket == "UMPIRE'S CALL":
        decision = "UMPIRE'S CALL"
    else:
        decision = "NOT OUT"

    return {
        "pitching": pitching,
        "impact": impact,
        "wicket": wicket,
        "decision": decision,
        "pitch": {"x": pitch_x, "y": pitch_y},
        "impact_point": {"x": impact_x, "y": impact_y},
        "projection": {"x": projected_x, "y": stump_y}
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video"})

    file = request.files["video"]
    batsman_hand = request.form.get("hand", "Right")

    filename = "hawk_" + str(uuid.uuid4())[:8] + os.path.splitext(secure_filename(file.filename))[1]
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    print(f">>> Received video: {filename}, batsman_hand={batsman_hand}")
    _t_total = time.time()

    points = detect_ball_opencv(path)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result = calculate_quadratic_path(points, width, height, batsman_hand)

    print(f">>> Total /analyze time: {time.time() - _t_total:.2f}s, "
          f"points used: {len(points)}")

    return jsonify({
        "success": True,
        "video": url_for("static", filename="uploads/" + filename),
        "ball_points": points,
        "analysis": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
