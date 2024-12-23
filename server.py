import os
from pathlib import Path
from datetime import datetime
from collections import deque
import base64
import traceback
import copy
import threading
import torch
from multiprocessing import Process
import webbrowser
import time
import atexit
import signal

from dotenv import load_dotenv
import boto3

from strongsort.strong_sort import StrongSORT
from strongsort.utils.parser import YamlParser

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import cv2
import numpy as np

from shapely.geometry import Polygon, Point

from ultralytics import YOLO
from ultralytics import solutions
from ultralytics.utils.plotting import Annotator, colors

from db import create_region, update_log_event, update_density_event, get_regions, get_logs, get_active_log, get_active_density_event, get_region_count

load_dotenv('./.env')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reid_weights = Path("weights\osnet_x0_25_msmt17.pt")
tracker_config = r"strongsort\configs\strongsort.yaml"
cfg = YamlParser()
cfg.merge_from_file(tracker_config)
tracker = StrongSORT(reid_weights,
                     device,
                     False,
                     max_dist=cfg.strongsort.max_dist,
                     max_iou_dist=cfg.strongsort.max_iou_dist,
                     max_age=cfg.strongsort.max_age,
                     max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
                     n_init=cfg.strongsort.n_init,
                     nn_budget=cfg.strongsort.nn_budget
                     )

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('HACKATHON_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('HACKATHON_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = os.getenv('HACKATHON_BUCKET')

# Base Response
response = {
     "footfall_summary": {
        "total_footfall": 0,
        "zone_footfall": {},
    },
    "high_density_times": [],
    "heatmap_urls": []
}

DENSITY_THRESHOLD = 10 
STEP = 1

frames = deque(maxlen=1000)

process_thread = None
stop_event = threading.Event()

# --- Util
def upload_to_s3(file_obj, filename):
    try:
        s3_client.upload_fileobj(
            file_obj,
            BUCKET_NAME,
            filename,
            ExtraArgs={"ContentType": "image/png"}
        )
        print("Image uploaded")
        return f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{filename}"
    except Exception as e:
        raise AppError(f"Failed to upload image to S3: {str(e)}", status_code=500)

# --- Flask
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Routes
@app.route("/process_stream", methods=["POST"])
def process_stream():
    global process_thread, stop_event

    data = request.json
    video_stream_url = data.get("video_stream_url", 0)  
    zones = data["zones"]

    if video_stream_url == "CAMERA":
        video_stream_url = 0

    if process_thread and process_thread.is_alive():
        return jsonify({"error": "Processing is already running"}), 400

    try:
        stop_event.clear()

        process_thread = threading.Thread(target=process_and_stream, args=(video_stream_url, zones))
        process_thread.daemon = True
        process_thread.start()

        return jsonify({"message": "Processing started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stop_process", methods=["POST"])
def stop_process():
    global stop_event

    if not process_thread or not process_thread.is_alive():
        return jsonify({"error": "No processing is currently running"}), 400

    stop_event.set()
    process_thread.join()

    response = {
        "footfall_summary": {
            "total_footfall": 0,
            "zone_footfall": {},
        },
        "high_density_times": [],
        "heatmap_urls": []
    }

    frames.clear()

    return jsonify({"message": "Processing stopped"}), 200

def process_and_stream(video_stream_url, zones):
    model = YOLO("yolo11m.pt")
    
    classes = [0]
    line_thickness = 2 
    region_thickness = 2 

    names = model.model.names
    videocapture = cv2.VideoCapture(video_stream_url)

    if not videocapture.isOpened():
        raise FileNotFoundError(f"Unable to open video stream: {video_stream_url}")

    for zone in zones:
        created_region = create_region(
            zone["zone_id"],
            zone["coordinates"]["x_min"], 
            zone["coordinates"]["y_min"], 
            zone["coordinates"]["x_max"], 
            zone["coordinates"]["y_max"]
        )
    
    regions = get_regions()
    REGION_COLORS = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in regions]

    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    object_paths = {}
    frame_counter = 0

    # Warm up the tracker model if needed
    if hasattr(tracker.model, 'warmup'):
        tracker.model.warmup()

    while videocapture.isOpened() and not stop_event.is_set():
        success, frame = videocapture.read()
        if not success:
            break

        regions = get_regions()

        frame_counter += 1 
    
        if frame_counter % STEP == 0:  
            frames.append(frame.copy())
            save_path = f"frames/frame_{frame_counter}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True) 
            cv2.imwrite(save_path, frame) 

        # Run YOLO detection
        results = model(frame, classes=classes)
        
        frame_time = datetime.now()

        # Process detections with StrongSORT
        if len(results[0].boxes) > 0:
            detections = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Filter for person class (usually class 0)
            mask = classes == 0
            detections = detections[mask]
            confidences = confidences[mask]
            classes = classes[mask]
            
            # Combine detections, confidences, and classes into a single array
            dets = np.hstack((detections, confidences[:, np.newaxis], classes[:, np.newaxis]))
            
            # Update tracker
            tracks = tracker.update(dets, frame)
            
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            response["footfall_summary"]["total_footfall"] = len(tracks)

            # Process each track
            for track in tracks:
                box = track[:4]  # First 4 elements are bounding box coordinates
                obj_id = int(track[4])  # Track ID
                cls = int(track[5])  # Class ID
                
                if cls == 0:  # If person class
                    annotator.box_label(box, str(obj_id), color=colors(cls, True))
                    bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                    # Track object paths
                    if obj_id not in object_paths:
                        object_paths[obj_id] = []
                    object_paths[obj_id].append(bbox_center)

                    # Process regions
                    for i, region in enumerate(regions):
                        region_color = (0, 0, 255)
                        cv2.rectangle(
                            frame,
                            (region["x_min"], region["y_min"]),
                            (region["x_max"], region["y_max"]),
                            region_color,
                            thickness=region_thickness
                        )

                        point = Point(bbox_center[0], bbox_center[1])
                        polygon = Polygon([
                            (region["x_min"], region["y_min"]),
                            (region["x_max"], region["y_min"]),
                            (region["x_max"], region["y_max"]),
                            (region["x_min"], region["y_max"])
                        ])

                        log = get_active_log(region['name'], obj_id)
                        
                        # ENTRY/EXIT Logic
                        if point.within(polygon):
                            if not log:
                                update_log_event(region["name"], obj_id)
                        else:
                            if log:
                                update_log_event(region["name"], obj_id)

        # Encode and emit frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        socketio.emit("frame", frame_base64)

        # Update response with region information
        for region in regions:
            region_count = get_region_count(region["name"])
            response["footfall_summary"]["zone_footfall"][region["name"]] = region_count

            event = get_active_density_event(region["name"])

            if not event and region_count >= DENSITY_THRESHOLD:
                update_density_event(region["name"])
            elif event and region_count < DENSITY_THRESHOLD:
                response["high_density_times"].append(update_density_event(region["name"]))

        socketio.emit("response", response)

        if stop_event.is_set():
            print("Stopping processing")
            break
    
    videocapture.release()

@app.route("/create_zone", methods=["POST"])
def create_zone():
    if not process_thread or not process_thread.is_alive():
        return jsonify({"error": "No processing is currently running"}), 400
    
    data = request.json
    created_region = create_region(
            data["zone_id"],
            data["coordinates"]["x_min"], 
            data["coordinates"]["y_min"], 
            data["coordinates"]["x_max"], 
            data["coordinates"]["y_max"]
        )

    return jsonify({"message": "Zone created"}), 200

@app.route("/generate_heatmap", methods=["GET"])
def generate_heatmap():
    copied_frames = copy.copy(frames)
    if not copied_frames:
        return jsonify({"error": "No frames available for heatmap generation."}), 400

    try:
        heatmap = solutions.Heatmap(
            model="yolo11m.pt",
            colormap=cv2.COLORMAP_PARULA,
            classes=[0],
            show=False
        )

        grayscale_frames = [] 

        print("Generating heatmap from frames...")
        for frame in copied_frames:
            heatmap_frame = heatmap.generate_heatmap(frame)
            gray_frame = cv2.cvtColor(heatmap_frame, cv2.COLOR_BGR2GRAY)
            grayscale_frames.append(gray_frame)

        print("Calculating average heatmap...")
        frame_stack = np.array(grayscale_frames, dtype=np.float32)
        average_heatmap = np.mean(frame_stack, axis=0)

        print("Normalizing heatmap...")
        normalized_heatmap = (average_heatmap - np.min(average_heatmap)) / \
        (np.max(average_heatmap) - np.min(average_heatmap))

        print("Saving average heatmap image...")
        heatmap_image = np.uint8(normalized_heatmap * 255)
        heatmap_color = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_VIRIDIS)
        filepath = f"./heatmaps/{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.png"
        cv2.imwrite(filepath, heatmap_color)

        s3_url = ""
        print("Uploading heatmap...")
        with open(filepath, "rb") as file_obj:
                s3_url = upload_to_s3(file_obj, os.path.basename(filepath))
        
        print("Updating response")
        response["heatmap_urls"].append(s3_url)
        
        return jsonify({"heatmap_url": s3_url}), 200
    except Exception as e:
        raise AppError(str(e), status_code=500)

# --- Error Handling
class AppError(Exception):
    def __init__(self, message, status_code=400):
        print(message)
        super().__init__(message)
        self.status_code = status_code

@app.errorhandler(AppError)
def handle_app_error(error):
    # Capture the full traceback of the error
    traceback_str = traceback.format_exc()
    print(f"AppError: {traceback_str}")
    return jsonify({"error": str(error), "traceback": traceback_str}), error.status_code

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    # Capture the full traceback of the error
    traceback_str = traceback.format_exc()
    print(f"Unexpected Error: {traceback_str}")
    return jsonify({"error": "An unexpected error occurred", "traceback": traceback_str}), 500

# --- Socket Logic
@socketio.on("connect")
def connect():
    print("Client connected")

@socketio.on("disconnect")
def disconnect():
    print("Client disconnected")

def cleanup_dashboard(dashboard_process):
    if dashboard_process and dashboard_process.is_alive():
        dashboard_process.terminate()
        dashboard_process.join()
        print("Dashboard terminated")

def signal_handler(signum, frame):
    print("Signal received, cleaning up...")
    cleanup_dashboard(dashboard_process)
    exit(0)

def launch_dashboard():
    try:
        from dashboard import app as dashboard_app
        dashboard_app.run_server(debug=False, port=8050)
    except Exception as e:
        print(f"Error launching dashboard: {e}")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start dashboard in separate process
    dashboard_process = Process(target=launch_dashboard)
    dashboard_process.daemon = True
    dashboard_process.start()
    
    # Register cleanup
    atexit.register(cleanup_dashboard, dashboard_process)
    
    # Give dashboard time to start
    time.sleep(2)
    
    # Open dashboard in browser
    webbrowser.open('http://127.0.0.1:8050')
    
    try:
        # Start main server
        socketio.run(app, host="0.0.0.0", port=5920)
    except KeyboardInterrupt:
        print("Server shutdown requested")
    finally:
        cleanup_dashboard(dashboard_process)


if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
