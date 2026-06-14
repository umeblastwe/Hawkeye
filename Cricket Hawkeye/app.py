import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
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
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # --- TESTING RESPONSE ---
        # Yahan aapka hawkeye algorithm run hoga baad mein.
        # Abhi testing ke liye hum static success response bhej rahe hain.
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully for testing!',
            'received_settings': {
                'batsman_hand': f"{batsman_hand} Handed",
                'original_umpires_call': umpires_call,
                'tracking_status': 'Ready for processing'
            },
            'video_path': filepath
        })
        
    return jsonify({'error': 'Invalid file type. Upload MP4, AVI, or MOV.'}), 400

if __name__ == '__main__':
    app.run(debug=True)