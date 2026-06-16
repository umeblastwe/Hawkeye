import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO Model (Aap baad mein yahan "best.pt" apna custom cricket model dal sakte hain)
model = YOLO("yolov8n.pt") 

# ==========================================
# LIGHTWEIGHT KALMAN FILTER FOR SMOOTHING
# ==========================================
class SimpleKalman:
    def __init__(self):
        # Initial states
        self.q = 0.05  # Process noise covariance
        self.r = 1.0   # Measurement noise covariance
        self.x = 0.0   # Estimated value
        self.p = 1.0   # Estimation error covariance
        self.k = 0.0   # Kalman gain

    def update(self, measurement):
        # Prediction update
        self.p = self.p + self.q
        # Measurement update
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return int(self.x)

# ==========================================
# ADVANCED TRACKING WITH PERSISTENCE
# ==========================================
def detect_ball_professional(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    raw_positions = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # PRO TIP FIX: Har frame pe fresh detection ke bajaye Object Tracking Buffer use kiya byte-track ke sath
        # conf low rakha (0.10) jaisa aapne kaha taake speed/blur miss na ho
        results = model.track(frame, persist=True, conf=0.10, verbose=False)
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                
                # Class 32 (Sports Ball)
                if class_id == 32:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    
                    raw_positions.append({"frame": frame_number, "x": cx, "y": cy})
                    break # Pehli milti hui valid ball configuration pick karein

    cap.release()

    # --- SIMULATED KALMAN FILTER SYSTEM TO ERADICATE ZIG-ZAG ---
    if len(raw_positions) < 3:
        return raw_positions

    kf_x = SimpleKalman()
    kf_y = SimpleKalman()
    
    # Initialize seeds
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
# PROFESSIONAL HAWK-EYE QUADRATIC PREDICTION
# ==========================================
def calculate_quadratic_path(points, width, height):
    if len(points) < 5:
        return {
            "pitching": "UNKNOWN", "impact": "UNKNOWN", "wicket": "UNKNOWN", "decision": "NOT OUT"
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    # BOUNCE DETECTION OVERHAUL:
    # Sirf argmax(ys) lagane se impact zone bounce ban jata tha. 
    # Ab hum velocity vector inversion dhoondte hain (jahan y-axis ka trend change ho rha ho)
    dy = np.diff(ys)
    bounce_index = np.argmax(ys) # Baseline fallback
    for i in range(1, len(dy)):
        if dy[i-1] > 0 and dy[i] < 0: # Velocity direction inversion detected
            bounce_index = i
            break

    pitch_x = float(xs[bounce_index])
    pitch_y = float(ys[bounce_index])

    # Impact parameter adjustment
    impact_index = min(bounce_index + 2, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    # REAL HAWK-EYE GRAPH: 2nd-Degree Quadratic Fit ($y = ax^2 + bx + c$)
    # Hum ys par xs fit kar rahe hain taake forward tracking clean parabolic curvature render kare
    coeff = np.polyfit(ys, xs, 2)
    
    stump_y = height * 0.60 # Standard alignment ratio for stumps base axis
    projected_x = float(np.polyval(coeff, stump_y))
    
    # Gate limits
    stump_left = width * 0.465
    stump_right = width * 0.535

    if stump_left <= projected_x <= stump_right:
        decision = "OUT"
        wicket = "HITTING"
    else:
        decision = "NOT OUT"
        wicket = "MISSING"

    return {
        "pitching": "IN LINE", "impact": "IN LINE", "wicket": wicket, "decision": decision,
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
    # RENDER STORAGE WORKAROUND: Client cache issues se bachne ke liye safe rendering uuid tag string
    filename = "hawk_" + str(uuid.uuid4())[:8] + os.path.splitext(secure_filename(file.filename))[1]
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    points = detect_ball_professional(path)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result = calculate_quadratic_path(points, width, height)

    return jsonify({
        "success": True,
        "video": url_for("static", filename="uploads/" + filename),
        "ball_points": points,
        "analysis": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
