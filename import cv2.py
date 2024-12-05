import cv2
import numpy as np
from ultralytics import YOLO 
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """동영상에서 지정된 간격으로 프레임을 추출합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * frame_rate)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def segment_road_and_sidewalk(frame, segmentation_model):
    """세그먼테이션 모델을 사용하여 도로와 인도를 구분합니다."""

    result = segmentation_model(frame)
    segmented_frame = result.segmentation_map  
    return segmented_frame

def detect_traffic_light(frame, yolo_model):
    """YOLO 모델을 사용하여 신호등을 탐지합니다."""
    results = yolo_model(frame)
    traffic_light_status = "Unknown"
    
    for result in results:
        if result["class"] == "traffic_light": 
           
            x, y, w, h = result["bbox"]
            traffic_light = frame[y:y+h, x:x+w]
            avg_color = np.mean(traffic_light, axis=(0, 1))
            if avg_color[2] > avg_color[1]: 
                traffic_light_status = "Stop"
            else:
                traffic_light_status = "Go"
    return traffic_light_status

def detect_crosswalk(frame, yolo_model):
    """YOLO 모델을 사용하여 횡단보도를 탐지합니다."""
    results = yolo_model(frame)
    crosswalk_detected = False
    
    for result in results:
        if result["class"] == "crosswalk":
            crosswalk_detected = True
            break
    return crosswalk_detected

def analyze_video(video_path):
    """동영상을 분석하여 도로, 신호등, 횡단보도를 탐지합니다."""
  
    yolo_model = YOLO("yolov8n.pt")
    segmentation_model = None 
    
    output_dir = "frames"
    frame_count = extract_frames(video_path, output_dir)
    
    for i in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
        frame = cv2.imread(frame_path)
        
   
        if segmentation_model:
            segmented_frame = segment_road_and_sidewalk(frame, segmentation_model)
            cv2.imshow("Segmented", segmented_frame)
        
      
        traffic_light_status = detect_traffic_light(frame, yolo_model)
        print(f"Frame {i}: Traffic Light - {traffic_light_status}")
        
  
        crosswalk_detected = detect_crosswalk(frame, yolo_model)
        print(f"Frame {i}: Crosswalk - {'Detected' if crosswalk_detected else 'Not Detected'}")
    
    cv2.destroyAllWindows()


analyze_video(r"C:\Users\qoran\Videos\녹음 2024-12-05 141249.mp4")
