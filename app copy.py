from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ROTATED_FOLDER = 'rotated'
ANALYZED_FOLDER = 'analyzed'
MODEL_FOLDER = 'model'

# Set the upload and rotated folders
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROTATED_FOLDER'] = ROTATED_FOLDER

app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['ANALYZED_FOLDER'] = ANALYZED_FOLDER

# Ensure the upload and rotated folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ROTATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANALYZED_FOLDER'], exist_ok=True)

base_url = "https://kasfa.mwonya.com"

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
            'original_image': f"{base_url}/{original_path}",
            'rotated_image': f"{base_url}/{app.config['ROTATED_FOLDER']}/{rotated_filename}",
            'time_taken': time_taken
        }

        return jsonify(response), 200

@app.route('/rotated/<filename>')
def get_rotated_image(filename):
    return send_from_directory(app.config['ROTATED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def get_upload_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyzed/<filename>')
def get_analyzed_image(filename):
    return send_from_directory(app.config['ANALYZED_FOLDER'], filename)

@app.route('/malai_upload', methods=['POST'])
def mal_upload_file():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the original image
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'best.pt')

        file.save(original_path)

        # Load model
        model = YOLO(model_path)

        # Run inference on image
        results = model(original_path)  # results list

        # Save the Analyzed image
        analysed_image_filename = f"analyzed_{file.filename}"
        analyzed_path = os.path.join(app.config['ANALYZED_FOLDER'], analysed_image_filename)

        # Show the results
        for r in results:
            im_array = r.plot(conf=False, labels=True)  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            im.save(analyzed_path)  # save image

        end_time = time.time()
        time_taken = end_time - start_time

        # Prepare the response in JSON format
        response = {
            'original_image': f"{base_url}/{original_path}",
            'rotated_image': f"{base_url}/{app.config['ANALYZED_FOLDER']}/{analysed_image_filename}",
            'time_taken': time_taken
        }

        return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)