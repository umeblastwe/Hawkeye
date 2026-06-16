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
# IMPROVED BALL TRACKING ENGINE
# ===============================
def detect_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    positions = []
    frame_number = 0
    last_point = None

    # MOG2 Background Subtractor use kar rahe hain noise khatam karne ke liye
    backSub = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=40, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # Apply Background Subtraction
        fg_mask = backSub.apply(frame)
        
        # Clean the mask using Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            # Cricket ball ka size frame mein chota hota hai (Range optimize ki hai)
            if area < 3 or area > 150:
                continue

            x, y, w, h = cv2.boundingRect(c)
            ratio = w / h if h else 0
            if ratio < 0.5 or ratio > 2.0:
                continue

            cx = x + w // 2
            cy = y + h // 2

            # Distance constraint check (Ball achanak 100px jump nahi kar sakti)
            if last_point:
                dist = np.sqrt((cx - last_point[0])**2 + (cy - last_point[1])**2)
                if dist > 60:  # Distance filter strictly 60px kiya
                    continue

            score = area
            if score > best_score:
                best_score = score
                best = (cx, cy)

        if best:
            positions.append({"frame": frame_number, "x": best[0], "y": best[1]})
            last_point = best

    cap.release()

    # --- TRAJECTORY SMOOTHING (ZIGZAG KHATAM KARNE KE LIYE) ---
    if len(positions) > 5:
        frames = [p["frame"] for p in positions]
        xs = [p["x"] for p in positions]
        ys = [p["y"] for p in positions]
        
        # Polynomial Smoothing filter lagaya taake zigzag points line mein ajayein
        poly_x = np.polyfit(frames, xs, 2)
        poly_y = np.polyfit(frames, ys, 2)
        
        smoothed_positions = []
        for i in range(len(positions)):
            smoothed_positions.append({
                "frame": frames[i],
                "x": int(np.polyval(poly_x, frames[i])),
                "y": int(np.polyval(poly_y, frames[i]))
            })
        return smoothed_positions

    return positions

# ===============================
# TRAJECTORY ANALYSIS
# ===============================
def calculate_path(points, width, height):
    if len(points) < 5:
        return {
            "pitching": "UNKNOWN", "impact": "UNKNOWN", "wicket": "UNKNOWN", "decision": "NOT OUT"
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    # Bounce point (Cricket camera views mein bounce highest Y point ya inverted graph check karta hai)
    bounce_index = np.argmax(ys)
    pitch_x = float(xs[bounce_index])
    pitch_y = float(ys[bounce_index])

    impact_index = min(bounce_index + 4, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    coeff = np.polyfit(ys, xs, 2)
    stump_y = height * 0.55  # Adjusted for better stump alignment view

    projected_x = float(np.polyval(coeff, stump_y))
    stump_left = width * 0.47
    stump_right = width * 0.53

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
