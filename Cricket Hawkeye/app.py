import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_automated_hawk_eye(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25
    
    ball_points = []
    frame_idx = 0
    impact_frame = None
    
    # Advanced background model to extract moving white/red ball matrices
    fgbg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=35, detectShadows=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        # Isolate lower half focus grid (Where pitch action happens)
        h, w, _ = frame.shape
        roi = frame[int(h*0.2):int(h*0.9), int(w*0.3):int(w*0.8)]
        
        mask = fgbg.apply(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 4 < area < 250:  # Size threshold metrics for pixel selection
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    # Map coordinates back to full frame size matrix
                    cX = int(M["m10"] / M["m00"]) + int(w*0.3)
                    cY = int(M["m01"] / M["m00"]) + int(h*0.2)
                    ball_points.append({"frame": frame_idx, "x": cX, "y": cY})
                    break
        
        # Automatic algorithmic check for directional change velocity drops (Impact Marker)
        if len(ball_points) > 3 and not impact_frame:
            v1_y = ball_points[-1]["y"] - ball_points[-2]["y"]
            v2_y = ball_points[-2]["y"] - ball_points[-3]["y"]
            # Ball tracking sequence redirection pattern detection
            if v1_y <= 0 and v2_y > 0:
                impact_frame = frame_idx

    cap.release()
    
    # Fail-safe processing parameter windows
    if not impact_frame and len(ball_points) > 0:
        impact_frame = ball_points[-1]["frame"]
    elif not impact_frame:
        impact_frame = 45 # Fallback center frame coordinate indicator
        
    impact_time = impact_frame / fps
    
    # Generate predictive path mappings to draw over canvas layers
    return {
        "impact_time": round(impact_time, 2),
        "tracking_points": ball_points[-20:] if len(ball_points) > 0 else []
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video asset submitted'}), 400
        
    file = request.files['video']
    batsman_hand = request.form.get('batsman_hand', 'Right')
    
    if file.filename == '':
        return jsonify({'error': 'Filename pointer is empty'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Calculate dynamic pixel arrays
        telemetry = process_automated_hawk_eye(filepath)
        
        return jsonify({
            'success': True,
            'video_url': url_for('static', filename=f'uploads/{filename}'),
            'impact_time': telemetry['impact_time'],
            'points': telemetry['tracking_points'],
            'batsman_hand': batsman_hand
        })

if __name__ == '__main__':
    app.run(debug=True)
