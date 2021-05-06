import torch
import json
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/best.pt', force_reload=False)

model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for head, body....

# Images
img = Image.open('demo.jpg')
# Inference
results = model(img, size=640)

results.print()  #display prediction statstics
results.save()  #save image of prediction
results.show()  #show the img with preidction
predictions = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

for p in predictions:
    print(p)
