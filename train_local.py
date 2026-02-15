from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")

model.train(
	
	data = "Find_nail_clipper-2/data.yaml",
	epochs = 30,
	imgsz = 640,
	
	project = "Model_v1",
	name = "nail_clipper_detact",
	
	device = 'cpu'
)

print("---- Finish training model where : {os.getcwd()}----")