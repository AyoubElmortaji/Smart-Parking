from flask import Flask, render_template, request, jsonify
import parking_detection
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)


os.makedirs(os.path.join(app.static_folder, 'images'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        result = parking_detection.detect_parking_spots()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)