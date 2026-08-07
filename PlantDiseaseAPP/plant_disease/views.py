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
FALLBACK_WEIGHTS = f"yolov8{getattr(settings, 'MODEL_SIZE', 'n')}.pt"

# Live-feed tuning (overridable via Django settings / env vars).
INFERENCE_IMGSZ = getattr(settings, "MODEL_IMGSZ", 416)
FRAME_SKIP = max(1, getattr(settings, "MODEL_FRAME_SKIP", 3))

# Demo mode: serve the dashboard from a folder of images instead of a camera
# or drone (real inference still runs on each image). Set DEMO_IMAGES_DIR to a
# folder of jpg/png files; the dashboard then defaults to this source, so the
# app can be showcased without any hardware. Falls back to the sample images
# shipped in the repo (docs/demo_images/) when the env var is unset.
_DEFAULT_DEMO_DIR = Path(settings.BASE_DIR).parent / "docs" / "demo_images"
DEMO_IMAGES_DIR = getattr(settings, "DEMO_IMAGES_DIR", "") or (
    str(_DEFAULT_DEMO_DIR) if _DEFAULT_DEMO_DIR.is_dir() else ""
)
DEMO_IMAGES = (
    sorted(p for p in Path(DEMO_IMAGES_DIR).iterdir()
           if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if DEMO_IMAGES_DIR else []
)
DEFAULT_SOURCE = "demo" if DEMO_IMAGES else "webcam"
_demo_index = 0

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


def _detect(frame):
    """Run inference on a frame, returning [(name, conf, x1, y1, x2, y2), ...]."""
    detections = []
    results = get_model().predict(frame, imgsz=INFERENCE_IMGSZ, verbose=False)
    for result in results:
        for box in result.boxes:
            detections.append((
                class_name(int(box.cls[0])),
                float(box.conf[0]),
                *map(int, box.xyxy[0]),
            ))
    return detections


def _draw(frame_rgb, detections):
    """Draw detection boxes/labels on a frame (cheap, no inference)."""
    for name, conf, x1, y1, x2, y2 in detections:
        label = f"{name}: {conf:.2f}"
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    return frame_rgb


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
    'wheat bacterial leaf streak (black chaff)', 'wheat head scab', 'wheat leaf rust',
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
        logger.info("Drone connected and streaming")
        return True
    except Exception as e:
        logger.warning("Failed to initialize drone: %s", e)
        return False

def initialize_webcam():
    global webcam
    try:
        webcam = cv2.VideoCapture(0)
        if webcam.isOpened():
            logger.info("Successfully opened camera")
            return True
        webcam.release()
        logger.warning("Failed to open camera")
        return False
    except Exception as e:
        logger.warning("Failed to initialize webcam: %s", e)
        return False

def _read_frame(source_type):
    """Return the next frame for a source, or None if it is unavailable.

    All sources return BGR frames (the format ultralytics/OpenCV expect).
    """
    global webcam, drone, _demo_index
    if source_type == 'demo':
        if not DEMO_IMAGES:
            return None
        frame = cv2.imread(str(DEMO_IMAGES[_demo_index % len(DEMO_IMAGES)]))
        _demo_index += 1
        return frame
    if source_type == 'drone':
        if drone is None:
            return None
        # djitellopy decodes frames as RGB (PyAV to_image()); convert to BGR
        # for the OpenCV/ultralytics pipeline.
        return cv2.cvtColor(drone.get_frame_read().frame, cv2.COLOR_RGB2BGR)
    if source_type == 'webcam':
        if webcam is None:
            return None
        ret, frame = webcam.read()
        return frame if ret else None
    return None

def generate_frames(source_type='webcam', show_overlay=True):
    frame_counter = 0
    last_detections = []

    while True:
        frame_rgb = _read_frame(source_type)
        if frame_rgb is None:
            time.sleep(0.1)  # Add small delay before next attempt
            continue

        if show_overlay:
            # Run YOLO inference only every FRAME_SKIP frames to keep the live
            # feed responsive; reuse the last detections for frames in between.
            if frame_counter % FRAME_SKIP == 0:
                last_detections = _detect(frame_rgb)
            frame_rgb = _draw(frame_rgb, last_detections)
            frame_counter += 1

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_rgb)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_analysis(source_type='webcam'):
    frame_counter = 0
    last_detections = []

    while True:
        frame_rgb = _read_frame(source_type)
        if frame_rgb is None:
            time.sleep(0.1)  # Add small delay before next attempt
            continue

        # Run inference only every FRAME_SKIP frames; reuse between runs.
        if frame_counter % FRAME_SKIP == 0:
            last_detections = _detect(frame_rgb)
        frame_counter += 1

        # Initialize analysis data
        analysis_data = []
        for name, conf, *_ in last_detections:
            analysis_data.append({
                'name': name,
                'confidence': conf,
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
    source_type = request.GET.get('source', DEFAULT_SOURCE)
    show_overlay = request.GET.get('overlay', 'false').lower() == 'true'

    if source_type == 'demo' and not DEMO_IMAGES:
        return HttpResponse("Demo mode is not configured. Set DEMO_IMAGES_DIR "
                            "to a folder of images and restart.")
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
    source_type = request.GET.get('source', DEFAULT_SOURCE)

    if source_type == 'demo' and not DEMO_IMAGES:
        return HttpResponse("Demo mode is not configured. Set DEMO_IMAGES_DIR "
                            "to a folder of images and restart.")
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
    global webcam, drone
    source_type = request.GET.get('source', DEFAULT_SOURCE)

    if source_type == 'demo':
        if not DEMO_IMAGES:
            return JsonResponse(
                {'error': 'Demo mode is not configured. Set DEMO_IMAGES_DIR.'},
                status=400
            )
    elif source_type == 'drone':
        if drone is None:
            return JsonResponse(
                {'error': 'No drone available. Load the dashboard or a '
                          'feed with ?source=drone first.'},
                status=400
            )
    elif source_type == 'webcam':
        if webcam is None:
            return JsonResponse({'error': 'No camera available'}, status=400)

    frame = _read_frame(source_type)
    if frame is None:
        return JsonResponse({'error': 'Failed to capture image'}, status=400)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = CAPTURES_DIR / filename
    
    # Run YOLO inference on the frame (same resolution as the live feed so the
    # captured analysis matches what is drawn on screen).
    results = get_model().predict(frame, imgsz=INFERENCE_IMGSZ, verbose=False)
    
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
