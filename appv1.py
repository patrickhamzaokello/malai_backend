from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ROTATED_FOLDER = 'rotated'

# Set the upload and rotated folders
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROTATED_FOLDER'] = ROTATED_FOLDER

# Ensure the upload and rotated folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ROTATED_FOLDER'], exist_ok=True)

@app.route('/')
def hello_world():
    return "Pk's Machine Learning Api Server v0.1!"

def rotate_image(image_path, degrees):
    img = Image.open(image_path)
    rotated_img = img.rotate(degrees)
    return rotated_img

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the original image
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(original_path)

        # Rotate the image by 45 degrees
        rotated_img = rotate_image(original_path, 45)

        # Save the rotated image
        rotated_filename = f"rotated_{file.filename}"
        rotated_path = os.path.join(app.config['ROTATED_FOLDER'], rotated_filename)
        rotated_img.save(rotated_path)

        end_time = time.time()
        time_taken = end_time - start_time

        # Prepare the response in JSON format
        response = {
            'original_image': f"{request.base_url}/{original_path}",
            'rotated_image': f"{request.base_url}/{app.config['ROTATED_FOLDER']}/{rotated_filename}",
            'time_taken': time_taken
        }

        return jsonify(response), 200

@app.route('/rotated/<filename>')
def get_rotated_image(filename):
    return send_from_directory(app.config['ROTATED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def get_upload_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)