import os
import random
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_lbw_analysis(batsman_hand, bowling_type, umpire_call,
                     pitch_line, impact_line, impact_height, projection_line, playing_shot):
    """
    Full LBW rule engine — Law 36 of Cricket.
    All inputs are string labels from the frontend.
    Returns a detailed verdict dict.
    """

    # ── Map string inputs to numeric values ──────────────────────────────────
    pitch_map = {
        'outside-leg': -2,
        'on-leg': -1,
        'middle-leg': -0.3,
        'middle': 0,
        'middle-off': 0.3,
        'on-off': 1,
        'outside-off': 2
    }
    impact_map = {
        'outside-leg': -2,
        'leg-stump': -1,
        'between-leg-mid': -0.4,
        'middle-stump': 0,
        'between-mid-off': 0.4,
        'off-stump': 1,
        'outside-off': 2
    }
    height_map = {
        'ankle': 0.05,
        'shin': 0.25,
        'below-knee': 0.38,
        'knee': 0.5,
        'above-knee': 0.62,
        'thigh': 0.72,
        'borderline': 0.82,
        'over-stumps': 0.95
    }
    proj_map = {
        'missing-leg': -2,
        'clipping-leg': -0.55,
        'hitting-leg': -0.4,
        'hitting-middle-leg': -0.2,
        'hitting-middle': 0,
        'hitting-middle-off': 0.2,
        'hitting-off': 0.4,
        'clipping-off': 0.55,
        'missing-off': 2
    }

    pitch_val = pitch_map.get(pitch_line, 0)
    impact_val = impact_map.get(impact_line, 0)
    height_val = height_map.get(impact_height, 0.5)
    proj_val = proj_map.get(projection_line, 0)

    # For LHB: flip leg/off orientation
    if batsman_hand == 'Left':
        pitch_val = -pitch_val
        impact_val = -impact_val
        proj_val = -proj_val

    # ── LBW Rule checks ──────────────────────────────────────────────────────
    # 1. Pitched outside leg stump?
    pitched_outside_leg = pitch_val < -1.05

    # 2. Impact check — outside off stump?
    impact_outside_off = impact_val > 1.05
    impact_on_edge_off = 0.85 < impact_val <= 1.05
    impact_on_edge_leg = -1.05 <= impact_val < -0.85
    impact_in_line = -1.05 <= impact_val <= 1.05

    # 3. Height checks
    height_over = height_val >= 0.90
    height_borderline = 0.78 <= height_val < 0.90
    height_ok = height_val < 0.78

    # 4. Projection checks
    proj_missing_leg = proj_val < -0.6
    proj_clipping = (-0.6 <= proj_val < -0.45) or (0.45 < proj_val <= 0.6)
    proj_hitting = -0.45 <= proj_val <= 0.45
    proj_missing_off = proj_val > 0.6

    # ── Bowling type notes ───────────────────────────────────────────────────
    bowl_notes = {
        'pace': 'Pace delivery — straight trajectory applied.',
        'seam': 'Seam movement — could go either way off the pitch.',
        'off': ('Off-spin turns away from RHB toward off side.' if batsman_hand == 'Right'
                else 'Off-spin turns into LHB from off side.'),
        'leg': ('Leg-spin turns into RHB from off side.' if batsman_hand == 'Right'
                else 'Leg-spin turns away from LHB.'),
        'left-arm-spin': ('Left-arm spin (Chinaman) turns into RHB.' if batsman_hand == 'Right'
                          else 'Left-arm spin turns away from LHB.'),
        'left-arm-pace': ('Left-arm pace angles in to RHB — watch pitch line.' if batsman_hand == 'Right'
                          else 'Left-arm pace angles away from LHB.')
    }
    bowl_note = bowl_notes.get(bowling_type, '')

    # ── Verdict logic (Law 36) ────────────────────────────────────────────────
    checks = {
        'pitching': '',
        'impact': '',
        'wickets': '',
        'height': ''
    }

    verdict = 'OUT'
    verdict_reasons = []
    umpires_call_reasons = []

    # Rule 1 — Pitched outside leg
    if pitched_outside_leg:
        verdict = 'NOT OUT'
        verdict_reasons.append('Ball pitched outside leg stump — automatic Not Out (Law 36.3)')
        checks['pitching'] = 'OUTSIDE LEG'
    else:
        checks['pitching'] = 'IN LINE' if abs(pitch_val) <= 1.05 else 'OUTSIDE OFF'

    # Rule 2 — Impact outside off + playing a shot
    if impact_outside_off and playing_shot == 'yes' and verdict == 'OUT':
        verdict = 'NOT OUT'
        verdict_reasons.append('Impact outside off stump while playing a shot — Not Out')
        checks['impact'] = 'OUTSIDE OFF'
    elif impact_on_edge_off or impact_on_edge_leg:
        checks['impact'] = "UMPIRE'S CALL"
    elif impact_in_line:
        checks['impact'] = 'IN-LINE'
    else:
        checks['impact'] = 'OUTSIDE OFF'

    # Rule 3 — Height
    if height_over and verdict == 'OUT':
        verdict = 'NOT OUT'
        verdict_reasons.append('Ball projected to pass over top of stumps — Not Out')
        checks['height'] = 'OVER STUMPS'
    elif height_borderline:
        checks['height'] = "UMPIRE'S CALL"
        umpires_call_reasons.append('Height borderline')
    else:
        checks['height'] = 'BELOW TOP'

    # Rule 4 — Projection (wickets)
    if proj_missing_leg and verdict == 'OUT':
        verdict = 'NOT OUT'
        verdict_reasons.append('Ball projected to miss leg stump — Not Out')
        checks['wickets'] = 'MISSING LEG'
    elif proj_missing_off and verdict == 'OUT':
        verdict = 'NOT OUT'
        verdict_reasons.append('Ball projected to miss off stump — Not Out')
        checks['wickets'] = 'MISSING OFF'
    elif proj_clipping:
        checks['wickets'] = "UMPIRE'S CALL"
        umpires_call_reasons.append('Only clipping stumps')
    elif proj_hitting:
        checks['wickets'] = 'HITTING'
    else:
        checks['wickets'] = 'MISSING'

    # Umpire's Call upgrade
    if verdict == 'OUT' and (umpires_call_reasons or impact_on_edge_off or impact_on_edge_leg):
        # If original umpire said NOT OUT, it stays NOT OUT (benefit of doubt)
        if umpire_call == 'Not Out':
            verdict = "UMPIRE'S CALL — NOT OUT"
        else:
            verdict = "UMPIRE'S CALL — OUT"

    # Build display panels (matching broadcast style)
    panels = [
        {
            'label': 'WICKETS',
            'value': checks['wickets'],
            'color': _panel_color(checks['wickets'])
        },
        {
            'label': 'IMPACT',
            'value': checks['impact'],
            'color': _panel_color(checks['impact'])
        },
        {
            'label': 'PITCHING',
            'value': checks['pitching'],
            'color': _panel_color(checks['pitching'])
        }
    ]

    # Final clean verdict label
    if verdict == 'OUT':
        display_verdict = 'OUT'
        verdict_color = '#ef4444'
    elif verdict == 'NOT OUT':
        display_verdict = 'NOT OUT'
        verdict_color = '#22c55e'
    else:
        display_verdict = verdict
        verdict_color = '#f59e0b'

    if not verdict_reasons:
        verdict_reasons = [
            f'Pitched: {checks["pitching"]}',
            f'Impact: {checks["impact"]}',
            f'Wickets: {checks["wickets"]}',
        ]

    return {
        'success': True,
        'verdict': display_verdict,
        'verdict_color': verdict_color,
        'panels': panels,
        'checks': checks,
        'reasons': verdict_reasons,
        'bowl_note': bowl_note,
        'umpire_original': umpire_call,
        'batsman_hand': batsman_hand,
        'bowling_type': bowling_type,
        'umpires_call_triggered': bool(umpires_call_reasons),
        'umpires_call_reasons': umpires_call_reasons
    }


def _panel_color(value):
    v = value.upper()
    if 'MISSING' in v or 'OUTSIDE' in v or 'OVER' in v:
        return 'red'
    if "UMPIRE" in v or 'CALL' in v or 'CLIPPING' in v or 'BORDERLINE' in v:
        return 'amber'
    if 'HITTING' in v or 'IN LINE' in v or 'IN-LINE' in v or 'BELOW' in v:
        return 'green'
    if 'LEG' in v and 'MISSING' not in v:
        return 'green'
    return 'blue'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type. Upload MP4, AVI, MOV, or WebM.'}), 400

    filename = secure_filename(file.filename)
    # Add unique prefix to avoid clashes
    unique_name = f"{random.randint(10000,99999)}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    # Grab form params
    batsman_hand   = request.form.get('batsman_hand', 'Right')
    bowling_type   = request.form.get('bowling_type', 'pace')
    umpire_call    = request.form.get('umpire_call', 'Not Out')
    pitch_line     = request.form.get('pitch_line', 'middle')
    impact_line    = request.form.get('impact_line', 'middle-stump')
    impact_height  = request.form.get('impact_height', 'knee')
    proj_line      = request.form.get('projection_line', 'hitting-middle')
    playing_shot   = request.form.get('playing_shot', 'yes')

    result = run_lbw_analysis(
        batsman_hand, bowling_type, umpire_call,
        pitch_line, impact_line, impact_height, proj_line, playing_shot
    )
    result['video_url'] = '/' + filepath.replace('\\', '/')
    result['filename'] = unique_name
    return jsonify(result)


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
