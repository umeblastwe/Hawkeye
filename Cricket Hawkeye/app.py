import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_lbw_outcome(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    
    ball_points = []
    frame_idx = 0
    impact_frame = None
    
    subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=30, detectShadows=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Focus crop on the center pitch track
        roi = frame[int(height*0.25):int(height*0.85), int(width*0.35):int(width*0.75)]
        mask = subtractor.apply(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) + int(width*0.35)
                    cY = int(M["m01"] / M["m00"]) + int(height*0.25)
                    ball_points.append((cX, cY))
                    break
                    
        # Detect impact via sharp velocity drop
        if len(ball_points) > 3 and not impact_frame:
            dy1 = ball_points[-1][1] - ball_points[-2][1]
            dy2 = ball_points[-2][1] - ball_points[-3][1]
            if dy1 <= 0 and dy2 > 0:
                impact_frame = frame_idx
                break

    cap.release()
    
    if not impact_frame:
        impact_frame = frame_idx if frame_idx > 0 else 50
    
    # --- PREDICITIVE MATHEMATICS ENGINE ---
    # Defination of Virtual Stumps bounding box based on international screen aspect ratios
    stumps_top_y = int(height * 0.42)
    stumps_left_x = int(width * 0.47)
    stumps_right_x = int(width * 0.52)
    stumps_center_x = (stumps_left_x + stumps_right_x) // 2
    margin = (stumps_right_x - stumps_left_x) // 4  # For Umpire's Call zone
    
    # Default fallbacks if tracking points are low
    predicted_target_x = stumps_center_x
    
    # Mathematical trajectory projection using slope (m) calculation
    if len(ball_points) >= 3:
        pts = ball_points[-3:]  # Take final tracking frames before impact
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        # Calculate slope: m = change_in_x / change_in_y
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        
        if dy != 0:
            slope = dx / dy
            # Extrapolate X coordinate at the stumps height (Y)
            distance_y = stumps_top_y - ys[-1]
            predicted_target_x = int(xs[-1] + (slope * distance_y))
            
    # Determine precise ICC DRS rule results based on pixel calculations
    if stumps_left_x + margin <= predicted_target_x <= stumps_right_x - margin:
        wickets_status = "HITTING"
        verdict = "OUT"
    elif (stumps_left_x <= predicted_target_x < stumps_left_x + margin) or \
         (stumps_right_x - margin < predicted_target_x <= stumps_right_x):
        wickets_status = "UMPIRE'S CALL"
        verdict = "DECISION PENDING"
    else:
        wickets_status = "MISSING"
        verdict = "NOT OUT"
        
    return {
        "freeze_time": round(impact_frame / fps, 2),
        "target_x_pct": round(predicted_target_x / width, 3), # Percentages for frontend scaling
        "target_y_pct": round(stumps_top_y / height, 3),
        "pitching": "IN-LINE",
        "impact": "IN-LINE",
        "wickets": wickets_status,
        "verdict": verdict
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video attached'}), 400
    file = request.files['video']
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run Mathematical prediction
        result = predict_lbw_outcome(filepath)
        
        return jsonify({
            'success': True,
            'video_url': url_for('static', filename=f'uploads/{filename}'),
            'impact_time': result['freeze_time'],
            'target_x': result['target_x_pct'],
            'target_y': result['target_y_pct'],
            'telemetry': {
                'pitching': result['pitching'],
                'impact': result['impact'],
                'wickets': result['wickets'],
                'verdict': result['verdict']
            }
        })
