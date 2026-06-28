from ultralytics import YOLO

model = YOLO("yolo26l.pt")

# Default tracker (BoT-SORT)
results = model.track(source="/home/takeshikato/data/VID_20260613_104929934.mp4", show=True)
