from roboflow import Roboflow

rf = Roboflow(api_key="WYS6zqFjqqkx3ecYCIfm")

project = rf.workspace("testobj-ollqp").project("find_nail_clipper")

dataset = project.version(2)

dataset.download("yolov11")

print("Dataset downloaded successfully!")