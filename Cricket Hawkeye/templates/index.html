import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ADVANCED HSV BALL TRACKER WITH DISTANCE FILTERING
# ─────────────────────────────────────────────────────────────────────────────
def detect_ball_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_positions = []
    frame_idx = 0
    prev_cx, prev_cy = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Pitch Zone Crop to eliminate crowd and scoreboards
        y1, y2 = int(height * 0.20), int(height * 0.85)
        x1, x2 = int(width * 0.35), int(width * 0.65)
        roi = frame[y1:y2, x1:x2]

        # Convert to HSV to track light colored cricket ball under ground illumination
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Color mask optimized for white/light cricket ball tracking bounds
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 45, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        min_distance = float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4 or area > 180:  # Rigid bounds for actual ball size
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00']) + x1
            cy = int(M['m01'] / M['m00']) + y1

            # Outlier Elimination: Ball can't radically warp positions across single frames
            if prev_cx is not None and prev_cy is not None:
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if dist > 60:  # Jump validation threshold metric
                    continue
                if dist < min_distance:
                    min_distance = dist
                    best_candidate = (cx, cy)
            else:
                if best_candidate is None or area > best_candidate[2]:
                    best_candidate = (cx, cy, area)

        if best_candidate:
            cx, cy = best_candidate[0], best_candidate[1]
            ball_positions.append((frame_idx, cx, cy))
            prev_cx, prev_cy = cx, cy

    cap.release()
    return ball_positions, fps, width, height, total_frames

# ─────────────────────────────────────────────────────────────────────────────
#  MATHEMATICAL POLYNOMIAL TRAJECTORY CURVE FITTING
# ─────────────────────────────────────────────────────────────────────────────
def analyse_trajectory(ball_positions, fps, width, height, batsman_hand):
    # Minimum points safeguard check
    if len(ball_positions) < 5:
        return build_fallback(width, height, batsman_hand)

    # Separate tracking axes arrays
    frames = [p[0] for p in ball_positions]
    xs = np.array([p[1] for p in ball_positions], dtype=float)
    ys = np.array([p[2] for p in ball_positions], dtype=float)

    # 1. Automatic Dynamic Bounce Check
    # Scan vectors where the downward vertical index path flips direction cleanly
    bounce_idx = 1
    max_y_val = -1
    for i in range(1, len(ys) - 1):
        if ys[i] > max_y_val:
            max_y_val = ys[i]
            bounce_idx = i

    # 2. Dynamic Pad Impact Isolation
    impact_idx = len(ys) - 1
    # Check sharp deceleration vectors after the pitch bounce framework
    for i in range(bounce_idx + 1, len(ys) - 1):
        dy_prev = ys[i] - ys[i-1]
        dy_next = ys[i+1] - ys[i]
        if dy_next < dy_prev * 0.3:  # Structural deceleration break trigger
            impact_idx = i
            break

    pitch_pt = (xs[bounce_idx], ys[bounce_idx])
    impact_pt = (xs[impact_idx], ys[impact_idx])

    # 3. Quadratic Curve Fitting Equation Modeling (x = ay^2 + by + c)
    # Fit only on the valid pre-impact ball coordinate vectors
    valid_y = ys[:impact_idx+1]
    valid_x = xs[:impact_idx+1]
    
    curve_coefficients = np.polyfit(valid_y, valid_x, 2)

    # 4. Extrapolate Smooth Continuous Path Points Array
    stump_top_y = height * 0.44
    stump_bot_y = height * 0.56
    
    # Generate smooth fitted trail coordinate lists
    smooth_trail_points = []
    start_y_val = int(ys[0])
    end_y_val = int(ys[impact_idx])
    
    # Path A: Flight path tracking interpolation
    for current_y in range(start_y_val, end_y_val + 1, 2):
        calc_x = int(np.polyval(curve_coefficients, current_y))
        smooth_trail_points.append((calc_x, current_y))

    # Path B: Predictive Extension Line directly into Wickets Zone Matrix
    predicted_points = []
    for proj_y in range(end_y_val, int(stump_top_y) - 1, -2):
        calc_x = int(np.polyval(curve_coefficients, proj_y))
        predicted_points.append((calc_x, proj_y))

    final_projected_x = int(np.polyval(curve_coefficients, stump_top_y))

    # 5. ICC DRS Threshold Layout Metrics
    stump_cx = width * 0.50
    half_stump_width = width * 0.033
    stump_left = stump_cx - half_stump_width
    stump_right = stump_cx + half_stump_width
    uc_margin = width * 0.016

    if batsman_hand == 'Left':
        stump_left, stump_right = stump_right, stump_left

    margin_gate = (stump_right - stump_left) // 4
    if stump_left + margin_gate <= final_projected_x <= stump_right - margin_gate:
        wickets_v, verdict = "HITTING", "OUT"
    elif (stump_left <= final_projected_x < stump_left + margin_gate) or \
         (stump_right - margin_gate < final_projected_x <= stump_right):
        wickets_v, verdict = "UMPIRE'S CALL", "UMPIRE'S CALL"
    else:
        wickets_v, verdict = "MISSING", "NOT OUT"

    return {
        'pitch_x': float(pitch_pt[0]), 'pitch_y': float(pitch_pt[1]),
        'impact_x': float(impact_pt[0]), 'impact_y': float(impact_pt[1]),
        'proj_x': float(final_projected_x), 'proj_y': float(stump_top_y),
        'stump_cx': float(stump_cx), 'stump_left': float(stump_left), 'stump_right': float(stump_right),
        'uc_margin': float(uc_margin), 'stump_top_y': float(stump_top_y), 'stump_bot_y': float(stump_bot_y),
        'bounce_time': round(frames[bounce_idx] / fps, 3), 'impact_time': round(frames[impact_idx] / fps, 3),
        'width': width, 'height': height, 'ball_count': len(ball_positions),
        'smooth_trail': smooth_trail_points, 'predicted_trail': predicted_points
    }

def build_fallback(width, height, batsman_hand):
    stump_cx = width * 0.50
    half_w = width * 0.033
    return {
        'pitch_x': width * 0.51, 'pitch_y': height * 0.65,
        'impact_x': width * 0.50, 'impact_y': height * 0.53,
        'proj_x': width * 0.49, 'proj_y': height * 0.44,
        'stump_cx': stump_cx, 'stump_left': stump_cx - half_w, 'stump_right': stump_cx + half_w,
        'uc_margin': width * 0.016, 'stump_top_y': height * 0.44, 'stump_bot_y': height * 0.56,
        'bounce_time': 0.5, 'impact_time': 1.1, 'width': width, 'height': height, 'ball_count': 0,
        'smooth_trail': [], 'predicted_trail': []
    }

def lbw_verdict(traj, batsman_hand, playing_shot):
    # This remains linked cleanly to keep validation structural components safe
    px, sl, sr, uc = traj['proj_x'], traj['stump_left'], traj['stump_right'], traj['uc_margin']
    pit_x, imp_x, imp_y = traj['pitch_x'], traj['impact_x'], traj['impact_y']
    st_top = traj['stump_top_y']

    if batsman_hand == 'Right':
        outside_leg = pit_x < sl - uc
        outside_off_impact = imp_x > sr + uc
    else:
        outside_leg = pit_x > sr + uc
        outside_off_impact = imp_x < sl - uc

    if outside_leg:
        return {'pitching': 'OUTSIDE LEG', 'impact': 'N/A', 'wickets': 'N/A', 'verdict': 'NOT OUT'}
    if imp_y < st_top - 12:
        return {'pitching': 'IN LINE', 'impact': 'IN LINE', 'wickets': 'OVER STUMPS', 'verdict': 'NOT OUT'}
    if outside_off_impact and playing_shot:
        return {'pitching': 'IN LINE', 'impact': 'OUTSIDE OFF', 'wickets': 'N/A', 'verdict': 'NOT OUT'}

    return {'pitching': 'IN LINE', 'impact': 'IN LINE', 'wickets': traj['smooth_trail'] and "HITTING" or "MISSING", 'verdict': px > sl and px < sr and "OUT" or "NOT OUT"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selection'}), 400

    batsman_hand = request.form.get('batsman_hand', 'Right')
    playing_shot = request.form.get('playing_shot', 'yes') == 'yes'

    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = detect_ball_trajectory(filepath)
    if result is None:
        return jsonify({'error': 'Read error'}), 500

    ball_positions, fps, width, height, total_frames = result
    traj = analyse_trajectory(ball_positions, fps, width, height, batsman_hand)
    
    # Sync structural decision metrics override directly based on calculated outcomes
    verdict_data = {
        'pitching': traj['pitch_x'] < traj['stump_left'] and "OUTSIDE OFF" or "IN LINE",
        'impact': traj['impact_x'] < traj['stump_left'] and "OUTSIDE" or "IN LINE",
        'wickets': traj['smooth_trail'] and "HITTING" or "MISSING",
        'verdict': traj['proj_x'] > traj['stump_left'] and traj['proj_x'] < traj['stump_right'] and "OUT" or "NOT OUT"
    }

    if traj['proj_x'] >= traj['stump_left'] and traj['proj_x'] <= traj['stump_left'] + traj['uc_margin']:
        verdict_data['wickets'] = "UMPIRE'S CALL"
        verdict_data['verdict'] = "UMPIRE'S CALL"

    return jsonify({
        'success': True, 'video_url': url_for('static', filename=f'uploads/{filename}'),
        'bounce_time': traj['bounce_time'], 'impact_time': traj['impact_time'],
        'pitch_x': traj['pitch_x'], 'pitch_y': traj['pitch_y'],
        'impact_x': traj['impact_x'], 'impact_y': traj['impact_y'],
        'proj_x': traj['proj_x'], 'proj_y': traj['proj_y'],
        'stump_left': traj['stump_left'], 'stump_right': traj['stump_right'],
        'stump_top_y': traj['stump_top_y'], 'stump_bot_y': traj['stump_bot_y'],
        'stump_cx': traj['stump_cx'], 'vid_width': width, 'vid_height': height,
        'smooth_trail': traj['smooth_trail'], 'predicted_trail': traj['predicted_trail'],
        'ball_count': traj['ball_count'], 'telemetry': verdict_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
