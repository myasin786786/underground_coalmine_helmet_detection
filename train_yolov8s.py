import ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO("/home/w.sabir/working_place/yasin/yolov8/models/yolov8s.pt")  # load a pretrained model (recommended for training)


# Save checkpoints after each epoch
model.train(
    data="/home/w.sabir/working_place/yasin/dataset/undergroud_coalmine_helmetdetection/data.yaml",
    name="yolov8_pretrained_small",
    epochs=200,
    imgsz=640,
    batch=16,
    #save_period=20,  # Save checkpoints every epoch
    device=[0,1,2,3],
    #resume=True,
    save=True,

)

