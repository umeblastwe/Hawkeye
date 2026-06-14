import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    
    # Telemetry simulation strings from user input
    pitching = request.form.get('pitching', 'In Line')
    impact = request.form.get('impact', 'In Line')
    wickets = request.form.get('wickets', 'Hitting')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # --- ICC DRS RULE ENGINE SIMULATION ---
        final_decision = "OUT"
        
        # Rule 1: Pitching outside leg is ALWAYS Not Out (regardless of umpire's call)
        if pitching == "Outside Leg":
            final_decision = "NOT OUT"
        
        # Rule 2: Impact outside off/stumps is Not Out UNLESS no stroke offered (assuming stroke played here)
        elif impact == "Outside":
            if umpires_call == "Out" and wickets == "Hitting":
                # Special fringe edge case evaluation
                final_decision = "NOT OUT"  # Standard stroke played scenario
            else:
                final_decision = "NOT OUT"
                
        # Rule 3: Missing is ALWAYS Not Out
        elif wickets == "Missing":
            final_decision = "NOT OUT"
            
        # Rule 4: Umpire's Call sticks to the original on-field decision
        elif wickets == "Umpire's Call":
            final_decision = "OUT" if umpires_call == "Out" else "NOT OUT"
            
        # Rule 5: If pitching in-line/off, impact in-line, and hitting -> It's plumb OUT
        elif wickets == "Hitting" and pitching in ["In Line", "Outside Off"] and impact == "In Line":
            final_decision = "OUT"

        # Web-accessible URL for the uploaded video file
        video_url = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify({
            'success': True,
            'video_url': video_url,
            'telemetry': {
                'batsman_stance': f"{batsman_hand} Handed",
                'pitching': pitching,
                'impact': impact,
                'wickets': wickets,
                'on_field_call': umpires_call,
                'final_decision': final_decision
            }
        })
        
    return jsonify({'error': 'Invalid file format.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
