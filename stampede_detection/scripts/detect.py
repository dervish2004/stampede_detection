import argparse
import os
import cv2
from ultralytics import YOLO

def detect_image(model, image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    results = model(img)
    annotated_img = results[0].plot()
    cv2.imwrite(output_path, annotated_img)
    print(f"Image detection done. Output saved to: {output_path}")
    cv2.imshow("YOLOv8 Detection - Image", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Processing video. This may take a while...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Detection - Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video detection interrupted by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video detection done. Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection: Image/Video")
    parser.add_argument('--model', type=str, default='models/yolov8n.pt', help='Path to YOLOv8 model weights')
    parser.add_argument('--type', type=str, choices=['image', 'video'], required=True, help='Detection type: image or video')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video')
    parser.add_argument('--output', type=str, default=None, help='Path to output annotated file')

    args = parser.parse_args()

    model = YOLO(args.model)

    # Set default output path if not provided
    base_out_dir = "results"
    os.makedirs(base_out_dir, exist_ok=True)
    in_filename = os.path.basename(args.input)
    if args.output is None:
        if args.type == "image":
            out_name = f"{os.path.splitext(in_filename)[0]}_annotated.jpg"
        else:
            out_name = f"{os.path.splitext(in_filename)[0]}_annotated.mp4"
        output_path = os.path.join(base_out_dir, out_name)
    else:
        output_path = args.output

    if args.type == "image":
        detect_image(model, args.input, output_path)
    else:
        detect_video(model, args.input, output_path)

if __name__ == "__main__":
    main()
