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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB


# ==========================================
# LIGHTWEIGHT KALMAN FILTER FOR SMOOTHING
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
# PURE OPENCV BALL TRACKING — no torch/YOLO needed
# Uses background subtraction + contour filtering to find the moving
# cricket ball. This uses only a few MB of RAM, fits Render Free plan.
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

    # Background subtractor — very lightweight compared to a neural net
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=20, varThreshold=28, detectShadows=False
    )

    raw_positions = []
    frame_number = 0
    start_time = time.time()

    # Restrict search to a central pitch-tracking zone — avoids crowd,
    # boundary boards, and sky from being misdetected as the ball.
    y1, y2 = int(height * 0.08), int(height * 0.92)
    x1, x2 = int(width * 0.18), int(width * 0.82)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if time.time() - start_time > max_seconds_budget:
            print(f">>> Time budget exceeded at frame {frame_number}, stopping early")
            break

        roi = frame[y1:y2, x1:x2]
        fg_mask = subtractor.apply(roi)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 6 or area > 500:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if aspect < 0.4 or aspect > 2.5:  # roughly circular/ball-shaped
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00']) + x1
            cy = int(M['m01'] / M['m00']) + y1
            if best is None or area > best[2]:
                best = (cx, cy, area)

        if best:
            raw_positions.append({"frame": frame_number, "x": best[0], "y": best[1]})

    cap.release()

    total_time = time.time() - start_time
    print(f">>> Processed {frame_number} frames in {total_time:.2f}s, "
          f"found {len(raw_positions)} ball detections")

    if len(raw_positions) < 3:
        return raw_positions

    kf_x = SimpleKalman()
    kf_y = SimpleKalman()
    kf_x.x = raw_positions[0]["x"]
    kf_y.x = raw_positions[0]["y"]

    smoothed_positions = []
    for p in raw_positions:
        smoothed_positions.append({
            "frame": p["frame"],
            "x": kf_x.update(p["x"]),
            "y": kf_y.update(p["y"])
        })

    return smoothed_positions


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

    # Find the bounce point — where the ball's vertical direction reverses
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

    # Fit a quadratic curve x = f(y) to project forward to stump height
    coeff = np.polyfit(ys, xs, 2)
    stump_y = height * 0.60
    projected_x = float(np.polyval(coeff, stump_y))

    stump_left = width * 0.465
    stump_right = width * 0.535
    uc_margin = width * 0.015

    # Pitching check — outside leg stump is an automatic Not Out
    leg_boundary  = stump_left  if batsman_hand == "Right" else stump_right
    off_boundary  = stump_right if batsman_hand == "Right" else stump_left
    pitched_outside_leg = (pitch_x < leg_boundary) if batsman_hand == "Right" else (pitch_x > leg_boundary)

    if pitched_outside_leg:
        pitching = "OUTSIDE LEG"
    elif (stump_left <= pitch_x <= stump_right):
        pitching = "IN LINE"
    else:
        pitching = "OUTSIDE OFF"

    impact_in_line = stump_left - uc_margin <= impact_x <= stump_right + uc_margin
    impact = "IN LINE" if impact_in_line else ("OUTSIDE OFF" if impact_x > stump_right else "OUTSIDE LEG")

    # Wickets / projection check (with Umpire's Call zone)
    if stump_left + uc_margin <= projected_x <= stump_right - uc_margin:
        wicket = "HITTING"
    elif (stump_left - uc_margin <= projected_x < stump_left + uc_margin) or \
         (stump_right - uc_margin < projected_x <= stump_right + uc_margin):
        wicket = "UMPIRE'S CALL"
    else:
        wicket = "MISSING"

    # ── Final verdict logic (Law 36) ──
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

    print(f">>> Total /analyze time: {time.time() - _t_total:.2f}s")

    return jsonify({
        "success": True,
        "video": url_for("static", filename="uploads/" + filename),
        "ball_points": points,
        "analysis": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
