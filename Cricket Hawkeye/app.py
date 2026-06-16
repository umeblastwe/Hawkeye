import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================================
# PROFESSIONAL BALL TRACKING ENGINE (HSV)
# ==========================================
def detect_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    positions = []
    frame_number = 0
    last_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # 1. BGR se HSV mein convert karein (Color isolation ke liye)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Cricket Ball ka color range (Standard broadcast mein white/light ball ke liye filter)
        # Yeh range sirf bright/moving cricket ball ko focus karegi
        lower_ball = np.array([0, 0, 200])
        upper_ball = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        
        # Noise saaf karne ke liye morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        min_dist = float('inf')

        for c in contours:
            area = cv2.contourArea(c)
            # Ball ka size screen par bohot chota (4px se 80px) hota hai
            if area < 3 or area > 80:
                continue

            x, y, w, h = cv2.boundingRect(c)
            
            # Ball hamesha round ya thodi oval hoti hai (Aspect Ratio Filter)
            aspect_ratio = float(w)/h
            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue

            cx = x + w // 2
            cy = y + h // 2

            # STRICT CONTINUITY FILTER: 
            # Ball achanak lambi chalang nahi maar sakti. Ek frame se dusre frame ka distance bohot kam hona chahiye.
            if last_point is not None:
                dist = np.sqrt((cx - last_point[0])**2 + (cy - last_point[1])**2)
                if dist < min_dist and dist < 35:  # Strict 35px threshold max movement per frame
                    min_dist = dist
                    best_candidate = (cx, cy)
            else:
                # Pehle frame mein ball hamesha upper half (pitch ke top area) se release hoti hai
                if cy < frame.shape[0] * 0.5:
                    best_candidate = (cx, cy)

        if best_candidate:
            positions.append({"frame": frame_number, "x": best_candidate[0], "y": best_candidate[1]})
            last_point = best_candidate

    cap.release()
    return positions

# ==========================================
# LINEAR PREDICTION & ANALYSIS
# ==========================================
def calculate_path(points, width, height):
    if len(points) < 4:
        return {
            "pitching": "UNKNOWN", "impact": "UNKNOWN", "wicket": "UNKNOWN", "decision": "NOT OUT"
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    # Bounce detection (Y-axis ka sabse niche wala max point)
    bounce_index = np.argmax(ys)
    pitch_x = float(xs[bounce_index])
    pitch_y = float(ys[bounce_index])

    impact_index = min(bounce_index + 2, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    # Linear regression baseline for straight trajectory path projection
    coeff = np.polyfit(ys, xs, 1)
    stump_y = height * 0.65  # Approximate stumps zone height

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
    app.run(host="0.0.0.0", port=5000)
