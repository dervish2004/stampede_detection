import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="yolov8n.yaml", help="YOLOv8 config or checkpoint")
    parser.add_argument('--data', type=str, default="config/data.yaml", help="Dataset config path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--project', type=str, default="results", help="Results directory")
    parser.add_argument('--name', type=str, default="stampede_train", help="Experiment name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)
