import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# ROBUST BALL TRACKING ENGINE
# ===============================
def detect_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    positions = []
    frame_number = 0
    last_point = None

    # MOG2 background segmenter for moving objects
    backSub = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=30, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # Isolate movement
        fg_mask = backSub.apply(frame)
        
        # Clean background noise using structural elements
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            # Size filters for standard cricket broadcast ball
            if area < 4 or area > 200:
                continue

            x, y, w, h = cv2.boundingRect(c)
            ratio = w / h if h else 0
            if ratio < 0.4 or ratio > 2.2:
                continue

            cx = x + w // 2
            cy = y + h // 2
            
            valid_candidates.append((cx, cy, area))

        # Tracking match logic
        best = None
        if valid_candidates:
            if last_point is None:
                # Agar pehla point hai to sabse relevant mass center select karein
                # Broadcasters videos mein upper half se ball release hoti hai
                valid_candidates.sort(key=lambda k: k[1]) 
                best = (valid_candidates[0][0], valid_candidates[0][1])
            else:
                # Pick the nearest neighbor to preserve sequential continuity
                min_dist = float('inf')
                for cx, cy, area in valid_candidates:
                    dist = np.sqrt((cx - last_point[0])**2 + (cy - last_point[1])**2)
                    # Ball movement frame-by-frame normal screen size par 75px se zyada jump nahi karti
                    if dist < min_dist and dist < 75:
                        min_dist = dist
                        best = (cx, cy)

        if best:
            positions.append({"frame": frame_number, "x": best[0], "y": best[1]})
            last_point = best

    cap.release()

    # --- NO-INVERSION SMOOTHING FILTER (Moving Average) ---
    if len(positions) > 3:
        smoothed = []
        window_size = 3
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            
            window_points = positions[start_idx:end_idx]
            avg_x = int(np.mean([p["x"] for p in window_points]))
            avg_y = int(np.mean([p["y"] for p in window_points]))
            
            smoothed.append({
                "frame": positions[i]["frame"],
                "x": avg_x,
                "y": avg_y
            })
        return smoothed

    return positions

# ===============================
# TRAJECTORY ANALYSIS ENGINE
# ===============================
def calculate_path(points, width, height):
    if len(points) < 4:
        return {
            "pitching": "UNKNOWN", "impact": "UNKNOWN", "wicket": "UNKNOWN", "decision": "NOT OUT"
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    # Dynamic Bounce Detection (Lowest point on screen is maximum Y coordinate)
    bounce_index = np.argmax(ys)
    pitch_x = float(xs[bounce_index])
    pitch_y = float(ys[bounce_index])

    # Impact calculation slightly post-bounce
    impact_index = min(bounce_index + 3, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    # Smooth parabolic projection for linear flow towards wickets
    coeff = np.polyfit(ys, xs, 1) # Linear fitting prevents inverse wrapping artifacts
    stump_y = height * 0.62  # Base/Middle alignment zone for stumps

    projected_x = float(np.polyval(coeff, stump_y))
    
    # Dynamic bounding box relative to camera center
    stump_left = width * 0.465
    stump_right = width * 0.535

    if stump_left <= projected_x <= stump_right:
        decision = "OUT"
        wicket = "HITTING"
    else:
        decision = "NOT OUT"
        wicket = "MISSING"

    return {
        "pitching": "IN LINE",
        "impact": "IN LINE",
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
    filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    points = detect_ball(path)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result = calculate_path(points, width, height)

    return jsonify({
        "success": True,
        "video": url_for("static", filename="uploads/" + filename),
        "ball_points": points,
        "analysis": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
