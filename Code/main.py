import inference


model = inference.get_roboflow_model("yolov9000")
results = model.infer(image="/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/expimg1.png")

# Extract object names (class_name) from predictions
object_names = [
    prediction.class_name
    for response in results
    for prediction in response.predictions
]

print("Object Names:", object_names)