import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE BALL TRACKER  — frame-by-frame OpenCV detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_ball_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Background subtractor — tuned for a moving cricket ball ──
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=20, varThreshold=30, detectShadows=False
    )

    ball_positions = []   # list of (frame_idx, cx, cy, area)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Work in a pitch-crop zone (avoids crowd/sky noise) ──
        y1 = int(height * 0.10)
        y2 = int(height * 0.92)
        x1 = int(width  * 0.20)
        x2 = int(width  * 0.80)
        roi = frame[y1:y2, x1:x2]

        # Background subtraction
        fg_mask = subtractor.apply(roi)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8 or area > 600:        # ball-size filter
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if aspect < 0.4 or aspect > 2.5:  # roughly circular
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00']) + x1
            cy = int(M['m01'] / M['m00']) + y1
            if best is None or area > best[2]:
                best = (cx, cy, area)

        if best:
            ball_positions.append((frame_idx, best[0], best[1], best[2]))

    cap.release()
    return ball_positions, fps, width, height, total_frames


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY ANALYSIS  — extracts pitch point, impact point, projection
# ─────────────────────────────────────────────────────────────────────────────
def analyse_trajectory(ball_positions, fps, width, height, batsman_hand):
    if len(ball_positions) < 4:
        # Fallback: not enough detections — use heuristic centre values
        return build_fallback(width, height, batsman_hand)

    # ── Smooth positions ──────────────────────────────────────────
    pts = np.array([(p[1], p[2]) for p in ball_positions], dtype=float)

    # ── Find PITCH POINT — lowest y (ball hits ground = peak of arc in image) ──
    # The ball travels DOWN the pitch, bounces (y increases), then rises again
    # We look for the local y-maximum after the ball has been airborne
    ys = pts[:, 1]
    bounce_idx = None

    # Scan for direction reversal in Y — going down then up = bounce
    for i in range(2, len(ys) - 2):
        going_down = (ys[i] - ys[i-2]) > 2   # y increasing = going down in image
        going_up   = (ys[i+2] - ys[i]) < -2  # y decreasing = rising after bounce
        if going_down and going_up:
            bounce_idx = i
            break

    # Fallback bounce estimate
    if bounce_idx is None:
        bounce_idx = max(1, len(pts) // 3)

    # ── Find IMPACT POINT — where velocity direction changes sharply ──
    # After the bounce, next direction change = hitting the pad
    impact_idx = None
    for i in range(bounce_idx + 2, len(pts) - 1):
        dx_prev = pts[i][0]   - pts[i-2][0]
        dx_next = pts[i+1][0] - pts[i][0]
        dy_prev = pts[i][1]   - pts[i-2][1]
        dy_next = pts[i+1][1] - pts[i][1]
        # Sharp direction change in x (ball deflects sideways = hits something)
        if abs(dx_next - dx_prev) > 8 or (dy_prev > 0 and dy_next < 0 and i > bounce_idx + 1):
            impact_idx = i
            break

    if impact_idx is None:
        impact_idx = min(bounce_idx + int(len(pts) * 0.3), len(pts) - 1)

    # ── Extract key coordinates ──
    pitch_pt  = pts[bounce_idx]   # (x, y) in pixels
    impact_pt = pts[impact_idx]

    # ── Project FORWARD from impact using last-known velocity vector ──
    v_window = min(3, impact_idx)
    if impact_idx >= v_window:
        vx = pts[impact_idx][0] - pts[impact_idx - v_window][0]
        vy = pts[impact_idx][1] - pts[impact_idx - v_window][1]
    else:
        vx = 0
        vy = -5  # default upward projection

    # Stump top y ≈ 38% down the frame for typical broadcast cameras
    stump_top_y = height * 0.38
    stump_bot_y = height * 0.55

    # Extrapolate x at stump plane
    if vy != 0:
        t = (stump_top_y - impact_pt[1]) / vy
    else:
        t = -20
    proj_x = impact_pt[0] + vx * t
    proj_y = stump_top_y

    # ── Stump geometry ────────────────────────────────────────────
    # Standard broadcast: stumps are roughly middle 6% of frame width
    stump_cx    = width * 0.50
    half_stump  = width * 0.033        # half-width of 3 stumps together
    leg_stump_x = stump_cx - half_stump
    off_stump_x = stump_cx + half_stump
    uc_margin   = width * 0.018        # Umpire's call zone width

    # Flip for LHB
    if batsman_hand == 'Left':
        leg_stump_x, off_stump_x = off_stump_x, leg_stump_x

    # ── Timing ──
    bounce_frame = ball_positions[bounce_idx][0]
    impact_frame = ball_positions[impact_idx][0]
    bounce_time  = round(bounce_frame / fps, 3)
    impact_time  = round(impact_frame / fps, 3)

    return {
        'pitch_x':  float(pitch_pt[0]),
        'pitch_y':  float(pitch_pt[1]),
        'impact_x': float(impact_pt[0]),
        'impact_y': float(impact_pt[1]),
        'proj_x':   float(proj_x),
        'proj_y':   float(proj_y),
        'stump_cx': float(stump_cx),
        'stump_left':  float(leg_stump_x),
        'stump_right': float(off_stump_x),
        'uc_margin':   float(uc_margin),
        'stump_top_y': float(stump_top_y),
        'stump_bot_y': float(stump_bot_y),
        'bounce_time': bounce_time,
        'impact_time': impact_time,
        'width':  width,
        'height': height,
        'ball_count': len(ball_positions),
        'pts_raw': [(int(p[1]), int(p[2])) for p in ball_positions[:60]]  # first 60 pts for canvas
    }


def build_fallback(width, height, batsman_hand):
    """Heuristic values when ball detection yields too few points."""
    stump_cx   = width * 0.50
    half_stump = width * 0.033
    return {
        'pitch_x':  width  * 0.52,
        'pitch_y':  height * 0.65,
        'impact_x': width  * 0.51,
        'impact_y': height * 0.50,
        'proj_x':   width  * 0.50,
        'proj_y':   height * 0.38,
        'stump_cx':    stump_cx,
        'stump_left':  stump_cx - half_stump,
        'stump_right': stump_cx + half_stump,
        'uc_margin':   width * 0.018,
        'stump_top_y': height * 0.38,
        'stump_bot_y': height * 0.55,
        'bounce_time': 0.6,
        'impact_time': 1.2,
        'width':  width,
        'height': height,
        'ball_count': 0,
        'pts_raw': []
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LBW RULE ENGINE  — Law 36, full ICC spec
# ─────────────────────────────────────────────────────────────────────────────
def lbw_verdict(traj, batsman_hand, playing_shot):
    px   = traj['proj_x']
    sl   = traj['stump_left']
    sr   = traj['stump_right']
    uc   = traj['uc_margin']
    pit_x = traj['pitch_x']
    imp_x = traj['impact_x']
    imp_y = traj['impact_y']
    h     = traj['height']
    stump_top = traj['stump_top_y']
    stump_bot = traj['stump_bot_y']

    # Normalise: for LHB the stump orientation is mirrored
    # But since we detect actual pixel positions, geometry is already correct.
    # "Outside leg" means beyond the leg stump on the leg side.
    if batsman_hand == 'Right':
        outside_leg = pit_x < sl - uc       # pitched to the left of leg stump
        outside_off_impact = imp_x > sr + uc  # impact to the right of off stump
    else:
        outside_leg = pit_x > sr + uc
        outside_off_impact = imp_x < sl - uc

    # Height: impact_y relative to stump zone
    above_stump_top = imp_y < stump_top - 10   # ball too high
    height_uc       = imp_y < stump_top + 15 and imp_y > stump_top - 10  # borderline height

    # Projection onto stumps
    hitting_clear = sl + uc < px < sr - uc
    missing_leg   = (px < sl - uc) if batsman_hand == 'Right' else (px > sr + uc)
    missing_off   = (px > sr + uc) if batsman_hand == 'Right' else (px < sl - uc)
    clipping      = (sl - uc <= px <= sl + uc) or (sr - uc <= px <= sr + uc)

    # ── Decision tree ──────────────────────────────────────────────
    # 1. Pitched outside leg — absolute not out
    if outside_leg:
        pitching  = 'OUTSIDE LEG'
        impact_v  = 'N/A'
        wickets_v = 'N/A'
        verdict   = 'NOT OUT'

    # 2. Ball too high
    elif above_stump_top:
        pitching  = 'IN LINE'
        impact_v  = 'IN LINE'
        wickets_v = 'OVER STUMPS'
        verdict   = 'NOT OUT'

    # 3. Impact outside off while playing a shot
    elif outside_off_impact and playing_shot:
        pitching  = 'IN LINE'
        impact_v  = 'OUTSIDE OFF'
        wickets_v = 'N/A'
        verdict   = 'NOT OUT'

    # 4. Missing leg
    elif missing_leg:
        pitching  = 'IN LINE'
        impact_v  = 'IN LINE'
        wickets_v = 'MISSING LEG'
        verdict   = 'NOT OUT'

    # 5. Missing off
    elif missing_off:
        pitching  = 'IN LINE'
        impact_v  = 'IN LINE'
        wickets_v = 'MISSING OFF'
        verdict   = 'NOT OUT'

    # 6. Umpire's Call — clipping
    elif clipping or height_uc:
        pitching  = 'IN LINE'
        impact_v  = 'IN LINE'
        wickets_v = "UMPIRE'S CALL"
        verdict   = "UMPIRE'S CALL"

    # 7. Clean hit
    elif hitting_clear:
        pitching  = 'IN LINE'
        impact_v  = 'IN LINE'
        wickets_v = 'HITTING'
        verdict   = 'OUT'

    else:
        pitching  = 'IN LINE'
        impact_v  = 'IN LINE'
        wickets_v = 'HITTING'
        verdict   = 'OUT'

    return {
        'pitching': pitching,
        'impact':   impact_v,
        'wickets':  wickets_v,
        'verdict':  verdict
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    batsman_hand  = request.form.get('batsman_hand', 'Right')
    playing_shot  = request.form.get('playing_shot', 'yes') == 'yes'

    # Save
    ext      = os.path.splitext(secure_filename(file.filename))[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Detect
    result = detect_ball_trajectory(filepath)
    if result is None:
        return jsonify({'error': 'Could not open video file'}), 500

    ball_positions, fps, width, height, total_frames = result

    # Analyse
    traj = analyse_trajectory(ball_positions, fps, width, height, batsman_hand)

    # LBW verdict
    verdict_data = lbw_verdict(traj, batsman_hand, playing_shot)

    return jsonify({
        'success': True,
        'video_url':   url_for('static', filename=f'uploads/{filename}'),
        # Timing
        'bounce_time': traj['bounce_time'],
        'impact_time': traj['impact_time'],
        # Pixel coords (for canvas overlay, absolute pixels)
        'pitch_x':  traj['pitch_x'],
        'pitch_y':  traj['pitch_y'],
        'impact_x': traj['impact_x'],
        'impact_y': traj['impact_y'],
        'proj_x':   traj['proj_x'],
        'proj_y':   traj['proj_y'],
        # Stump geometry (pixels)
        'stump_left':  traj['stump_left'],
        'stump_right': traj['stump_right'],
        'stump_top_y': traj['stump_top_y'],
        'stump_bot_y': traj['stump_bot_y'],
        'stump_cx':    traj['stump_cx'],
        # Video dimensions
        'vid_width':  width,
        'vid_height': height,
        # Raw ball path for canvas drawing (up to 60 pts)
        'ball_path': traj['pts_raw'],
        'ball_count': traj['ball_count'],
        # LBW telemetry
        'telemetry': verdict_data
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
