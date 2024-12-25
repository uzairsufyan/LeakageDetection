import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import uuid  # To generate unique IDs for tracked objects

import matplotlib.pyplot as plt
from pymongo import MongoClient

# Load the YOLO model
model = YOLO('C:/Users/Uzair Sufiyan/Downloads/pt/best.pt')  # Update path to your model

class_count = defaultdict(int)

client = MongoClient("mongodb+srv://<username>:<password>@cluster0.u4fe3.mongodb.net/")
db = client["event_database"]  # Database name
collection = db["event_logs1"]

# Tracking data
tracked_objects = {}
IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.35
MAX_FRAME_LOSS = 50  # Frames after which a tracked object is considered lost
DETECTION_TIME_LIMIT = 30 # in seconds

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def match_existing_objects(bbox, event_type):
    """Check if the bounding box matches an existing object."""
    for obj_id, obj_data in tracked_objects.items():
        if obj_data["event_type"] == event_type:
            iou = calculate_iou(bbox, obj_data["bbox"])
            if iou > IOU_THRESHOLD:
              
                # Update bounding box and reset frames_left

                obj_data["bbox"] = bbox
                obj_data["frames_left"] = MAX_FRAME_LOSS
                return obj_id  # Matched object ID
    return None

# Function to process video and return annotated frames and event logs
def process_video_realtime(input_video_path):
    # Open video file
    global class_count, tracked_objects

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    stframe = st.empty()  # Placeholder for displaying frames in real-time
    event_log_placeholder = st.empty()
    event_log = []  # Store timestamps of detections 

   

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number and timestamp
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model(frame_rgb)
        detections = results[0].boxes  # Extract detection results

        names = {0: 'bumpy area', 1: 'cylinder', 2: 'human activity', 3: 'leakage', 4: 'plug'}

        to_remove = []
        for obj_id, obj_data in tracked_objects.items():
            obj_data["frames_left"] -= 1
            if obj_data["frames_left"] <= 0:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del tracked_objects[obj_id]

        detected_events = []

        # Check for specific classes (human activity, leakage) - Update class IDs as needed
        for box in detections:
            class_id = int(box.cls)
            conf = float(box.conf)
            event_type = names.get(class_id, "Unknown Event")
           

            if event_type in ["human activity", "leakage", "cylinder"] and conf >= CONFIDENCE_THRESHOLD:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)

                # Check if this detection matches an existing object
                matched_obj_id = match_existing_objects(bbox, event_type)

                current_time = time.time()

                if not matched_obj_id:
                    # New detection
                    obj_id = str(uuid.uuid4())
                    tracked_objects[obj_id] = {"bbox": bbox, 
                                                "frames_left": MAX_FRAME_LOSS, 
                                                "event_type": event_type, 
                                                "last_detected": current_time}

                    # Log the detection
                    if event_type in ["human activity", "leakage"]:
                        event = {
                        "event_type": event_type,
                        "timestamp": timestamp
                        }
                        collection.insert_one(event)  # Insert event into MongoDB
                        event_log.append(f"{event_type} detected at {timestamp}")
                        class_count[event_type] += 1
                        detected_events.append(event_type)
                    elif event_type == "cylinder":
                        class_count[event_type]+= 1
                else:
                    last_detection_time = tracked_objects[matched_obj_id]["last_detected"]
                    time_diff = current_time - last_detection_time

                    if time_diff > DETECTION_TIME_LIMIT:
                        if event_type in ["human activity", "leakage"]:
                            event = {
                            "event_type": event_type,
                            "timestamp": timestamp
                                    }
                            collection.insert_one(event)  # Insert event into MongoDB
                            event_log.append(f"{event_type} detected at {timestamp}")
                            class_count[event_type] += 1
                            detected_events.append(event_type)
                        elif event_type == "cylinder":
                            class_count[event_type] += 1
                        # Update the matched object with the new detection if the time difference is greater than 20 seconds
                    tracked_objects[matched_obj_id]["bbox"] = bbox
                    tracked_objects[matched_obj_id]["frames_left"] = MAX_FRAME_LOSS

                    # Update the last detection time for the matched object
                    tracked_objects[matched_obj_id]["last_detected"] = current_time

        annotated_frame = results[0].plot()

        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        stframe.image(annotated_frame_bgr, channels="BGR", use_container_width=True)

        event_log_placeholder.markdown("### Event Log")
        for event in event_log[-10:]:  # Limit to last 10 events for better readability
            event_log_placeholder.write(event)

        

    cap.release()
    return event_log

# Streamlit UI
st.title("YOLO Real-Time Video Analysis")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
        tmp_video_file.write(uploaded_video.read())
        input_video_path = tmp_video_file.name

    # Display the uploaded video preview
    st.video(input_video_path)

    # Process the video in real time and log events
    if st.button("Run YOLO Detection"):
        with st.spinner("Processing video..."):
            event_log = process_video_realtime(input_video_path)

        # Display the event log
        st.subheader("Full Event Log")
        for event in event_log:
            st.write(event)

        filtered_counts = {key: value for key, value in class_count.items() if key in ['human activity', 'leakage']}

        # Create a DataFrame for plotting
        df = pd.DataFrame(list(filtered_counts.items()), columns=["Class", "Count"])

        fig, ax = plt.subplots()

        # Plot vertical bars with custom width
        ax.bar(df['Class'], df['Count'], width=0.4)  # Set the width of the bars here (e.g., 0.4)

        # Set the labels and title
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Counts for Human Activity & Leakage')

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.subheader("Class Detection Counts:")
        for event_type, count in class_count.items():
            st.write(f"{count} {event_type} detected")
    
