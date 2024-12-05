import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    saved_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps * interval) == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames saved in {output_dir}")

def main(video_path, output_dir):
    extract_frames(video_path, output_dir)

    for frame_file in sorted(os.listdir(output_dir)):
        frame_path = os.path.join(output_dir, frame_file)
        frame = cv2.imread(frame_path)

       

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/zeonons/Documents/opencv.mp4"  
    output_dir = "frames"         
    main(video_path, output_dir)
