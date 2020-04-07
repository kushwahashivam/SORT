import os
import time
import numpy as np
import cv2
import torch

from models import Darknet
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# frame_size = (704, 1280)
frame_size = (480, 640)
img_size = (256, 416)  # (320, 192) or (416, 256) or (608, 352) for (height, width)
conf_thres = .3
nms_thres = .5
cfg = "cfg/yolov3-tiny.cfg"
weights = "weights/yolov3-tiny.pt"
# source = "Mujhko Barsaat Bana Lo.MP4"
source = 1

model = Darknet(cfg, img_size).to(device)
if os.path.exists(weights):
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
else:
    raise FileNotFoundError(weights + " not found.")
model.eval()
# model.half()

if torch.cuda.is_available() and type(source) is str:
    torch.backends.cudnn.benchmark = True  # speed up constant image size inference
classes = load_classes("coco.names")
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
frame_transform = lambda x: x[:frame_size[0], :frame_size[1]]
img_transform = lambda x: cv2.resize(x, (img_size[1], img_size[0]))

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error opening video stream.")
while cap.isOpened():
    t1 = time.time()
    ret, frame = cap.read()
    frame = frame_transform(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print(frame.shape)
    if ret == True:
        with torch.no_grad():
            img = (img_transform(frame) / 255.).astype(np.float32)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            det, _ = model(img)
            det = non_max_suppression(det, conf_thres, nms_thres)
            det = det[0]
            # print(type(det))
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_size, det[:, :4], frame_size)
                det = det.cpu().numpy()
                for *xyxy, conf, _, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detections", frame)
        t2 = time.time()
        print("fps: ", 1/(t2-t1))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()