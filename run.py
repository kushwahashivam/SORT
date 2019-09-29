import os
import time
import cv2
import torch
import numpy as np
import torchvision as tv
from scipy.optimize import linear_sum_assignment

from models import Darknet
from tracker import *
from utils.utils import *
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # speed up constant image size inference

tracking_list = []

model = Darknet(cfg, img_size).to(device)
if os.path.exists(weights):
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
else:
    raise FileNotFoundError(weights + " not found.")
model.eval()

classes = load_classes("coco.names")
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
frame_transform = lambda x: x[:frame_size[0], :frame_size[1]]
img_transform = lambda x: cv2.resize(x, (img_size[1], img_size[0]))

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error opening video stream.")

while cap.isOpened():
    # t1 = time.time()
    ret, frame = cap.read()
    if ret == True:
        frame = frame_transform(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = []
        non_overlaps = []
        with torch.no_grad():
            img = (img_transform(frame) / 255.).astype(np.float32)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            pred, _ = model(img)
            pred = non_max_suppression(pred, conf_thres, nms_thres)
            pred = pred[0]
            if pred is not None and len(pred):
                pred[:, :4] = scale_coords(img_size, pred[:, :4], frame_size)
                pred = pred.cpu().numpy()
                for *xyxy, conf, _, cls in pred:
                    if cls == 0:
                        detections.append(xyxy)
                detections = np.array(detections, dtype=np.float32)
                if len(tracking_list) != 0:
                    tracks = [(t.predict()[0] if t.is_mature() else t.get_state()) for t in tracking_list]
                    tracks = torch.from_numpy(np.array(tracks, dtype=np.float32))
                    ious = tv.ops.box_iou(torch.from_numpy(detections), tracks).numpy()
                    ious *= ious >= iou_thres
                    row_ind, col_ind = linear_sum_assignment(-ious)
                    for i in range(len(detections)):
                        if i in row_ind:
                            ind, = np.where(row_ind == i)
                            col = int(col_ind[ind])
                            if ious[i, col] > 0.:
                                tracking_list[col].update(detections[i])
                                tracking_list[col].detected = True  # To be used for displaying detected ones
                                continue
                        tracking_list.append(KalmanObjectTracker(detections[i]))
                        tracking_list[-1].color = colors[np.random.randint(low=0, high=len(colors))]
                else:
                    for i in range(len(detections)):
                        tracking_list.append(KalmanObjectTracker(detections[i]))
                        tracking_list[-1].color = colors[np.random.randint(low=0, high=len(colors))]
        new_tl = tracking_list[:]
        for tracker in tracking_list:
            if tracker.is_mature() and not tracker.expired() and tracker.detected:
                try:
                    xyxy, lbl, color = tracker.get_state(), str(tracker.id), tracker.color
                    plot_one_box(xyxy, frame, label=lbl, color=color)
                except ValueError:
                    print("NaN occured.")
                    new_tl.remove(tracker)
            elif tracker.expired():
                new_tl.remove(tracker)
            tracker.detected = False  # To be used for displaying detected ones
        tracking_list = new_tl
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detections", frame)
        # t2 = time.time()
        # print("fps: ", 1/(t2-t1))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
