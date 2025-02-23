from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from pathlib import Path
from django.conf import settings
import cv2
import time
import json
import logging
from datetime import datetime
import zipfile
import io

logger = logging.getLogger(__name__)

# Model weights - use a trained model if present, otherwise fall back to a
# pretrained YOLO model so the app works out of the box.
WEIGHTS = Path(settings.BASE_DIR) / "plant_disease" / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

_model = None


def get_model():
    """Load the detection model lazily so the app can boot without weights."""
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    if WEIGHTS.exists():
        _model = YOLO(WEIGHTS)
        logger.info("Loaded custom model weights from %s", WEIGHTS)
    else:
        logger.warning(
            "%s not found - falling back to %s. Place your trained model at that "
            "path for best results.", WEIGHTS, FALLBACK_WEIGHTS
        )
        _model = YOLO(FALLBACK_WEIGHTS)
    return _model


def class_name(cls):
    """Resolve a class index to a human-readable name."""
    names = getattr(get_model(), "names", None)
    if names:
        name = names.get(cls)
        if name:
            return name
    if 0 <= cls < len(DISEASE_CLASSES):
        return DISEASE_CLASSES[cls]
    return f"class {cls}"

# Create captures directory if it doesn't exist
CAPTURES_DIR = Path(settings.BASE_DIR) / "plant_disease" / "static" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Store capture history
capture_history = []

# Disease class names
DISEASE_CLASSES = [
    'apple black rot', 'apple mosaic virus', 'apple rust', 'apple scab',
    'banana anthracnose', 'banana black leaf streak', 'banana bunchy top',
    'banana cigar end rot', 'banana cordana leaf spot', 'banana panama disease',
    'basil downy mildew', 'bean halo blight', 'bean mosaic virus', 'bean rust',
    'bell pepper bacterial spot', 'bell pepper blossom end rot',
    'bell pepper frogeye leaf spot', 'bell pepper powdery mildew',
    'blueberry anthracnose', 'blueberry botrytis blight', 'blueberry mummy berry',
    'blueberry rust', 'blueberry scorch', 'broccoli alternaria leaf spot',
    'broccoli downy mildew', 'broccoli ring spot', 'cabbage alternaria leaf spot',
    'cabbage black rot', 'cabbage downy mildew', 'carrot alternaria leaf blight',
    'carrot cavity spot', 'carrot cercospora leaf blight',
    'cauliflower alternaria leaf spot', 'cauliflower bacterial soft rot',
    'celery anthracnose', 'celery early blight', 'cherry leaf spot',
    'cherry powdery mildew', 'citrus canker', 'citrus greening disease',
    'coffee berry blotch', 'coffee black rot', 'coffee brown eye spot',
    'coffee leaf rust', 'corn gray leaf spot', 'corn northern leaf blight',
    'corn rust', 'corn smut', 'cucumber angular leaf spot',
    'cucumber bacterial wilt', 'cucumber powdery mildew',
    'eggplant cercospora leaf spot', 'eggplant phomopsis fruit rot',
    'eggplant phytophthora blight', 'garlic leaf blight', 'garlic rust',
    'ginger leaf spot', 'ginger sheath blight', 'grape black rot',
    'grape downy mildew', 'grape leaf spot', 'grapevine leafroll disease',
    'lettuce downy mildew', 'lettuce mosaic virus', 'maple tar spot',
    'peach anthracnose', 'peach brown rot', 'peach leaf curl', 'peach rust',
    'peach scab', 'plum bacterial spot', 'plum brown rot', 'plum pocket disease',
    'plum pox virus', 'plum rust', 'potato early blight', 'potato late blight',
    'raspberry fire blight', 'raspberry gray mold', 'raspberry leaf spot',
    'raspberry yellow rust', 'rice blast', 'rice sheath blight',
    'soybean bacterial blight', 'soybean brown spot', 'soybean downy mildew',
    'soybean frog eye leaf spot', 'soybean mosaic', 'soybean rust',
    'squash powdery mildew', 'strawberry anthracnose', 'strawberry leaf scorch',
    'tobacco blue mold', 'tobacco brown spot', 'tobacco frogeye leaf spot',
    'tobacco mosaic virus', 'tomato bacterial leaf spot', 'tomato early blight',
    'tomato late blight', 'tomato leaf mold', 'tomato mosaic virus',
    'tomato septoria leaf spot', 'tomato yellow leaf curl virus',
    'wheat bacterial leaf streak', 'wheat head scab', 'wheat leaf rust',
    'wheat loose smut', 'wheat powdery mildew', 'wheat septoria blotch',
    'wheat stem rust', 'wheat stripe rust', 'zucchini bacterial wilt',
    'zucchini downy mildew', 'zucchini powdery mildew',
    'zucchini yellow mosaic virus'
]

# Initialize video sources
drone = None
webcam = None

def initialize_drone():
    global drone
    from djitellopy import Tello
    try:
        drone = Tello()
        drone.connect()
        drone.streamon()
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Failed to initialize drone: {e}")
        return False

def initialize_webcam():
    global webcam
    try:
        webcam = cv2.VideoCapture(0)
        if webcam.isOpened():
            print("Successfully opened camera")
            return True
        webcam.release()
        print("Failed to open camera")
        return False
    except Exception as e:
        print(f"Failed to initialize webcam: {e}")
        return False

def generate_frames(source_type='webcam', show_overlay=True):
    global drone, webcam
    
    while True:
        frame = None
        
        if source_type == 'drone' and drone is not None:
            frame = drone.get_frame_read().frame
            # For drone, convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif source_type == 'webcam' and webcam is not None:
            ret, frame = webcam.read()
            if not ret:
                time.sleep(0.1)  # Add small delay before next attempt
                continue
            frame_rgb = frame
        
        if frame is None:
            time.sleep(0.1)  # Add small delay before next attempt
            continue

        if show_overlay:
            # Run YOLO inference on the frame
            results = get_model()(frame_rgb)

            # Process results and add disease names
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class index and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get disease name from class index
                    disease_name = class_name(cls)
                    
                    # Add label with disease name and confidence
                    label = f"{disease_name}: {conf:.2f}"
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw box and label with professional font
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_rgb)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_analysis(source_type='webcam'):
    global drone, webcam
    
    while True:
        frame = None
        
        if source_type == 'drone' and drone is not None:
            frame = drone.get_frame_read().frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif source_type == 'webcam' and webcam is not None:
            ret, frame = webcam.read()
            if not ret:
                time.sleep(0.1)  # Add small delay before next attempt
                continue
            frame_rgb = frame
        
        if frame is None:
            time.sleep(0.1)  # Add small delay before next attempt
            continue

        # Initialize analysis data
        analysis_data = []

        # Run YOLO inference on the frame
        results = get_model()(frame_rgb)

        # Process results and add disease names
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class index and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get disease name from class index
                disease_name = class_name(cls)
                
                # Add to analysis data
                analysis_data.append({
                    'name': disease_name,
                    'confidence': float(conf)  # Convert to float for JSON serialization
                })

        # If no diseases detected, add healthy tissue
        if not analysis_data:
            analysis_data.append({
                'name': 'Healthy Tissue',
                'confidence': 1.0
            })

        # Sort by confidence
        analysis_data.sort(key=lambda x: x['confidence'], reverse=True)

        # Send the analysis data
        yield f"data: {json.dumps(analysis_data)}\n\n"
        time.sleep(0.5)  # Update every 500ms

def get_disease_description(disease_name):
    descriptions = {
        'tomato late blight': 'Fungal disease causing dark lesions on leaves and stems. Requires immediate treatment.',
        'potato late blight': 'Serious fungal disease affecting leaves and tubers. Can spread rapidly in wet conditions.',
        'apple scab': 'Fungal disease causing dark, scabby lesions on leaves and fruit.',
        'grape leaf spot': 'Fungal disease causing circular spots on leaves. Can affect fruit quality.',
        # Add more descriptions as needed
    }
    return descriptions.get(disease_name.lower(), 'Plant disease detected. Monitor for symptoms and consider treatment if necessary.')

def index(request):
    return render(request, 'plant_disease/index.html')

def video_feed(request):
    source_type = request.GET.get('source', 'webcam')
    show_overlay = request.GET.get('overlay', 'false').lower() == 'true'
    
    if source_type == 'drone' and drone is None:
        if not initialize_drone():
            return HttpResponse("Failed to connect to drone")
    elif source_type == 'webcam' and webcam is None:
        if not initialize_webcam():
            return HttpResponse("Failed to connect to webcam. Please check if a camera is connected and accessible.")
    
    return StreamingHttpResponse(
        generate_frames(source_type, show_overlay),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def analysis_feed(request):
    source_type = request.GET.get('source', 'webcam')
    
    if source_type == 'drone' and drone is None:
        if not initialize_drone():
            return HttpResponse("Failed to connect to drone")
    elif source_type == 'webcam' and webcam is None:
        if not initialize_webcam():
            return HttpResponse("Failed to connect to webcam")
    
    return StreamingHttpResponse(
        generate_analysis(source_type),
        content_type='text/event-stream'
    )

def cleanup():
    global drone, webcam
    if drone is not None:
        drone.streamoff()
        drone.end()
    if webcam is not None:
        webcam.release()

def capture_image(request):
    global webcam
    if webcam is None:
        return JsonResponse({'error': 'No camera available'}, status=400)
    
    ret, frame = webcam.read()
    if not ret:
        return JsonResponse({'error': 'Failed to capture image'}, status=400)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = CAPTURES_DIR / filename
    
    # Run YOLO inference on the frame
    results = get_model()(frame)
    
    # Draw detection boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class index and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get disease name from class index
            disease_name = class_name(cls)
            
            # Add label with disease name and confidence
            label = f"{disease_name}: {conf:.2f}"
            
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    
    # Save the image with detection boxes
    cv2.imwrite(str(filepath), frame)
    
    # Prepare analysis data
    analysis_data = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            disease_name = class_name(cls)
            analysis_data.append({
                'name': disease_name,
                'confidence': float(conf),
                'description': get_disease_description(disease_name)
            })
    
    if not analysis_data:
        analysis_data.append({
            'name': 'Healthy Tissue',
            'confidence': 1.0,
            'description': 'No diseases detected in the current view.'
        })
    
    # Add to history
    capture_info = {
        'timestamp': timestamp,
        'filename': filename,
        'analysis': analysis_data
    }
    capture_history.append(capture_info)
    
    return JsonResponse({
        'success': True,
        'filename': filename,
        'analysis': analysis_data
    })

def get_history(request):
    return JsonResponse({'history': capture_history})

def export_results(request):
    if not capture_history:
        return JsonResponse({'error': 'No captures to export'}, status=400)
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add each captured image to the zip
        for capture in capture_history:
            image_path = CAPTURES_DIR / capture['filename']
            if image_path.exists():
                # Read the image file
                with open(image_path, 'rb') as f:
                    # Add to zip with timestamp in filename
                    zip_file.writestr(
                        f"capture_{capture['timestamp']}.jpg",
                        f.read()
                    )
    
    # Prepare the response
    zip_buffer.seek(0)
    response = HttpResponse(
        zip_buffer.getvalue(),
        content_type='application/zip'
    )
    
    # Set the filename in the Content-Disposition header
    filename = f"plant_disease_captures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    response['Content-Length'] = len(zip_buffer.getvalue())
    
    return response
