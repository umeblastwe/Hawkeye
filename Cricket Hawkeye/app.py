import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_cricket_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ball_coords = []
    frame_count = 0
    impact_frame = None
    
    # Background Subtractor ball detection ke liye
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Masking out unwanted areas (focusing only on pitch region)
        mask = fgbg.apply(frame)
        
        # Finding contours of moving objects (The Ball)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_frame_coords = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 300:  # Ball size detection bounds
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    current_frame_coords = (cX, cY)
                    ball_coords.append((frame_count, cX, cY))
                    break
        
        # Automated Impact Detection: Agar ball sudden direction change kare ya stop ho
        if len(ball_coords) > 3 and not impact_frame:
            # Calculating velocity vector differentials
            dy1 = ball_coords[-1][2] - ball_coords[-2][2]
            dy2 = ball_coords[-2][2] - ball_coords[-3][2]
            
            # Agar vertical movement drastically reverse ya drop ho jaye (Pad Impact)
            if dy1 >= 0 and dy2 < 0:  
                impact_frame = frame_count
                
    cap.release()
    
    # Default fallbacks agar CV anomalies detect na karein
    if not impact_frame:
        impact_frame = int(frame_count * 0.6)  # Statistical safe middle ground
        
    # Standard tracking coordinates calculation relative to video matrix
    impact_timestamp = impact_frame / fps if fps > 0 else 2.5
    
    return {
        "impact_time": round(impact_timestamp, 2),
        "resolution": {"width": width, "height": height},
        "total_frames": frame_count
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    file = request.files['video']
    batsman_hand = request.form.get('batsman_hand', 'Right')
    umpires_call = request.form.get('umpires_call', 'Not Out')
    
    # Telemetry criteria selection matrices
    pitching = request.form.get('pitching', 'IN-LINE')
    impact_pos = request.form.get('impact', 'IN-LINE')
    wickets = request.form.get('wickets', 'HITTING')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run OpenCV Tracking Computation Engine
        cv_results = analyze_cricket_trajectory(filepath)
        
        # Decision Matrix evaluation logic
        final_decision = "OUT"
        if pitching == "OUTSIDE LEG" or impact_pos == "OUTSIDE" or wickets == "MISSING":
            final_decision = "NOT OUT"
        elif wickets == "UMPIRE'S CALL":
            final_decision = "OUT" if umpires_call == "Out" else "NOT OUT"
            
        return jsonify({
            'success': True,
            'video_url': url_for('static', filename=f'uploads/{filename}'),
            'impact_time': cv_results['impact_time'],
            'telemetry': {
                'pitching': pitching,
                'impact': impact_pos,
                'wickets': wickets,
                'on_field_call': umpires_call,
                'final_decision': final_decision
            }
        })

if __name__ == '__main__':
    app.run(debug=True)
