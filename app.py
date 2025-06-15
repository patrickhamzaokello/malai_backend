from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from ultralytics import YOLO
import os
import time
import json

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

def extract_detection_data(results):
    """Extract detailed information from YOLO detection results with malaria-specific analysis"""
    detections = []
    summary = {
        'total_objects': 0,
        'class_counts': {},
        'confidence_stats': {
            'average': 0,
            'min': 1,
            'max': 0
        }
    }
    
    all_confidences = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                # Extract box coordinates (xyxy format)
                box_coords = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = box_coords
                
                # Calculate width, height, and area
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Get confidence score
                confidence = float(boxes.conf[i])
                all_confidences.append(confidence)
                
                # Get class ID and name
                class_id = int(boxes.cls[i])
                class_name = r.names[class_id]
                
                # Update class counts
                if class_name in summary['class_counts']:
                    summary['class_counts'][class_name] += 1
                else:
                    summary['class_counts'][class_name] = 1
                
                # Create detection object with malaria-specific analysis
                detection = {
                    'id': i,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': round(confidence, 3),
                    'confidence_level': get_confidence_level(confidence),
                    'bounding_box': {
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2),
                        'width': round(width, 2),
                        'height': round(height, 2),
                        'area': round(area, 2)
                    },
                    'center_point': {
                        'x': round((x1 + x2) / 2, 2),
                        'y': round((y1 + y2) / 2, 2)
                    },
                    'parasite_size': categorize_parasite_size(area)
                }
                
                detections.append(detection)
    
    # Calculate summary statistics
    summary['total_objects'] = len(detections)
    
    if all_confidences:
        summary['confidence_stats']['average'] = round(sum(all_confidences) / len(all_confidences), 3)
        summary['confidence_stats']['min'] = round(min(all_confidences), 3)
        summary['confidence_stats']['max'] = round(max(all_confidences), 3)
    
    return detections, summary

def get_confidence_level(confidence):
    """Categorize confidence levels for medical interpretation"""
    if confidence >= 0.9:
        return "very_high"
    elif confidence >= 0.8:
        return "high"
    elif confidence >= 0.7:
        return "medium"
    elif confidence >= 0.6:
        return "low"
    else:
        return "very_low"

def categorize_parasite_size(area):
    """Categorize parasite size based on bounding box area"""
    if area < 300:
        return "small"
    elif area < 800:
        return "medium"  
    elif area < 1500:
        return "large"
    else:
        return "very_large"

def generate_lab_analysis(detections, image_metadata):
    """Generate laboratory analysis report for technicians"""
    total_parasites = len(detections)
    
    if total_parasites == 0:
        return {
            'test_result': 'NEGATIVE',
            'parasite_count': 0,
            'parasitemia_estimation': 'Not detected',
            'parasitemia_percentage': 0,
            'lab_notes': 'No P. falciparum parasites detected in blood smear. Consider manual microscopy verification if clinical suspicion remains high.',
            'quality_control': {
                'image_quality': assess_image_quality(image_metadata),
                'detection_reliability': 'N/A - No detections',
                'review_needed': False
            },
            'technician_action': 'Report negative result. Archive sample images.'
        }
    
    # Calculate parasitemia (simplified estimation for lab reporting)
    image_area = image_metadata['width'] * image_metadata['height']
    estimated_field_count = image_area / 10000  # Approximate microscopic fields
    parasites_per_field = round(total_parasites / estimated_field_count, 2) if estimated_field_count > 0 else total_parasites
    
    # Determine parasitemia level for lab classification
    if total_parasites <= 2:
        parasitemia_level = 'Low'
        priority = 'routine'
    elif total_parasites <= 10:
        parasitemia_level = 'Moderate'
        priority = 'standard'
    else:
        parasitemia_level = 'High'
        priority = 'urgent'
    
    # Quality control metrics
    high_confidence_count = sum(1 for d in detections if d['confidence'] >= 0.8)
    medium_confidence_count = sum(1 for d in detections if 0.6 <= d['confidence'] < 0.8)
    low_confidence_count = sum(1 for d in detections if d['confidence'] < 0.6)
    
    confidence_ratio = high_confidence_count / total_parasites if total_parasites > 0 else 0
    
    # Determine if manual review is needed
    review_needed = confidence_ratio < 0.5 or low_confidence_count > 2
    
    # Lab-specific confidence assessment
    if confidence_ratio >= 0.8:
        detection_reliability = 'High - Automated result reliable'
    elif confidence_ratio >= 0.6:
        detection_reliability = 'Good - Standard verification recommended'
    else:
        detection_reliability = 'Low - Manual microscopy verification required'
    
    # Generate lab notes
    lab_notes = f'P. falciparum parasites detected: {total_parasites} organisms identified. '
    lab_notes += f'Density classification: {parasitemia_level}. '
    if review_needed:
        lab_notes += 'ATTENTION: Some detections have lower confidence scores - recommend manual verification.'
    else:
        lab_notes += 'Detection confidence acceptable for automated reporting.'
    
    # Technician action recommendations
    if priority == 'urgent':
        technician_action = 'URGENT: High parasite density detected. Verify result and expedite report to clinician.'
    elif review_needed:
        technician_action = 'Manual microscopy verification recommended before final reporting.'
    else:
        technician_action = 'Standard processing - prepare report for pathologist review.'
    
    return {
        'test_result': 'POSITIVE',
        'parasite_count': total_parasites,
        'parasitemia_estimation': parasitemia_level,
        'parasites_per_field': parasites_per_field,
        'lab_notes': lab_notes,
        'priority_level': priority,
        'quality_control': {
            'image_quality': assess_image_quality(image_metadata),
            'detection_reliability': detection_reliability,
            'confidence_breakdown': {
                'high_confidence': high_confidence_count,
                'medium_confidence': medium_confidence_count,
                'low_confidence': low_confidence_count
            },
            'review_needed': review_needed,
            'confidence_ratio': round(confidence_ratio, 3)
        },
        'technician_action': technician_action,
        'morphology_data': {
            'total_detections': total_parasites,
            'average_parasite_size': round(sum(d['bounding_box']['area'] for d in detections) / total_parasites, 2) if total_parasites > 0 else 0,
            'size_distribution': get_size_distribution(detections)
        }
    }

def assess_image_quality(metadata):
    """Assess image quality for lab QC purposes"""
    width, height = metadata['width'], metadata['height']
    file_size = metadata['size_bytes']
    
    # Basic quality assessment
    resolution_score = 'Good' if width >= 400 and height >= 300 else 'Low'
    file_size_score = 'Good' if file_size > 20000 else 'Compressed'
    
    if resolution_score == 'Good' and file_size_score == 'Good':
        return 'Excellent - Suitable for automated analysis'
    elif resolution_score == 'Good':
        return 'Good - Adequate for analysis'
    else:
        return 'Poor - Consider higher resolution imaging'

def get_size_distribution(detections):
    """Get parasite size distribution for morphological analysis"""
    if not detections:
        return {}
    
    sizes = [d['parasite_size'] for d in detections]
    distribution = {}
    for size in ['small', 'medium', 'large', 'very_large']:
        distribution[size] = sizes.count(size)
    
    return distribution

def get_image_metadata(image_path):
    """Extract image metadata"""
    with Image.open(image_path) as img:
        return {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'size_bytes': os.path.getsize(image_path)
        }

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

    # Get optional parameters
    confidence_threshold = float(request.form.get('confidence', 0.25))
    save_image = request.form.get('save_image', 'true').lower() == 'true'

    if file:
        try:
            # Save the original image
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            model_path = os.path.join(app.config['MODEL_FOLDER'], 'best.pt')

            file.save(original_path)

            # Get image metadata
            image_metadata = get_image_metadata(original_path)

            # Load model
            model = YOLO(model_path)

            # Run inference on image with confidence threshold
            results = model(original_path, conf=confidence_threshold)

            # Extract detailed detection data
            detections, summary = extract_detection_data(results)
            
            # Generate lab analysis report
            lab_analysis = generate_lab_analysis(detections, image_metadata)

            analyzed_image_url = None
            if save_image:
                # Save the analyzed image
                analyzed_image_filename = f"analyzed_{file.filename}"
                analyzed_path = os.path.join(app.config['ANALYZED_FOLDER'], analyzed_image_filename)

                # Show the results
                for r in results:
                    im_array = r.plot(conf=False, labels=True)  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    im.save(analyzed_path)  # save image

                analyzed_image_url = f"{base_url}/{app.config['ANALYZED_FOLDER']}/{analyzed_image_filename}"

            end_time = time.time()
            time_taken = round(end_time - start_time, 3)

            # Prepare the enhanced response with malaria-specific diagnosis
            response = {
                'rotated_image': analyzed_image_url, # for android
                'time_taken': time_taken, #for android
                'original_image': f"{base_url}/{original_path}", #for android
                'success': True,
                'processing_info': {
                    'time_taken': time_taken,
                    'confidence_threshold': confidence_threshold,
                    'model_info': {
                        'model_path': model_path,
                        'model_type': 'YOLO',
                        'medical_application': 'Malaria Parasite Detection',
                        'target_pathogen': 'Plasmodium falciparum'
                    }
                },
                'image_info': {
                    'original_filename': file.filename,
                    'original_image_url': f"{base_url}/{original_path}",
                    'analyzed_image_url': analyzed_image_url,
                    'metadata': image_metadata,
                    'sample_type': 'blood_smear'
                },
                'detection_results': {
                    'summary': summary,
                    'detections': detections
                },
                'laboratory_analysis': lab_analysis
            }

            return jsonify(response), 200

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'processing_info': {
                    'time_taken': round(time.time() - start_time, 3)
                }
            }), 500

@app.route('/malai_upload_batch', methods=['POST'])
def mal_upload_batch():
    """Enhanced endpoint for batch processing multiple images"""
    start_time = time.time()
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    confidence_threshold = float(request.form.get('confidence', 0.25))
    save_images = request.form.get('save_images', 'true').lower() == 'true'
    
    batch_results = []
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'best.pt')
    model = YOLO(model_path)
    
    for file in files:
        if file.filename == '':
            continue
            
        try:
            file_start_time = time.time()
            
            # Save and process each file
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(original_path)
            
            image_metadata = get_image_metadata(original_path)
            results = model(original_path, conf=confidence_threshold)
            detections, summary = extract_detection_data(results)
            
            # Generate lab analysis report
            lab_analysis = generate_lab_analysis(detections, image_metadata)
            
            analyzed_image_url = None
            if save_images:
                analyzed_image_filename = f"analyzed_{file.filename}"
                analyzed_path = os.path.join(app.config['ANALYZED_FOLDER'], analyzed_image_filename)
                
                for r in results:
                    im_array = r.plot(conf=False, labels=True)
                    im = Image.fromarray(im_array[..., ::-1])
                    im.save(analyzed_path)
                
                analyzed_image_url = f"{base_url}/{app.config['ANALYZED_FOLDER']}/{analyzed_image_filename}"
            
            file_result = {
                'filename': file.filename,
                'success': True,
                'processing_time': round(time.time() - file_start_time, 3),
                'image_info': {
                    'original_image_url': f"{base_url}/{original_path}",
                    'analyzed_image_url': analyzed_image_url,
                    'metadata': image_metadata,
                    'sample_type': 'blood_smear'
                },
                'detection_results': {
                    'summary': summary,
                    'detections': detections
                },
                'laboratory_analysis': lab_analysis
            }
            
        except Exception as e:
            file_result = {
                'filename': file.filename,
                'success': False,
                'error': str(e),
                'processing_time': round(time.time() - file_start_time, 3)
            }
        
        batch_results.append(file_result)
    
    total_time = round(time.time() - start_time, 3)
    
    return jsonify({
        'success': True,
        'batch_info': {
            'total_files': len(files),
            'processed_files': len(batch_results),
            'total_processing_time': total_time,
            'confidence_threshold': confidence_threshold
        },
        'results': batch_results
    }), 200

if __name__ == '__main__':
    app.run(debug=True)