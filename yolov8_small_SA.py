import ultralytics
from ultralytics import YOLO

# Load a model
#model = YOLO("/home/w.sabir/working_place/yasin/yolov8_attention_mechanism/Coalmine_Detection_Improved_YOLOv8/ultralytics/cfg/models/v8/yolov8_ResBlock_CBAM.yaml").load("/home/w.sabir/working_place/yasin/yolov8_attention_mechanism/Coalmine_Detection_Improved_YOLOv8/runs/detect/yolov8_smallModel_with_custom_ResBlock_CBAM/weights/last.pt")  # load a custome ECA model 
model = YOLO("/home/w.sabir/working_place/yasin/yolov8_attention_mechanism/Coalmine_Detection_Improved_YOLOv8/ultralytics/cfg/models/v8/yolov8_SA.yaml").load("/home/w.sabir/working_place/yasin/yolov8/models/yolov8s.pt")  # load a custome ECA model 


# Save checkpoints after each epoch
model.train(
    data="/home/w.sabir/working_place/yasin/dataset/undergroud_coalmine_helmetdetection/data.yaml",
    name="yolov8_smallModel_with_custom_SA",
    epochs=200,
    imgsz=640,
    batch=16,
    #save_period=20,  # Save checkpoints every epoch
    device=[0,1,2,3],
    resume=True,
    save=True,

)

