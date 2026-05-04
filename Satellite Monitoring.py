from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from layers.WHTResNet import wht_resnet50

HT_type = 1
video_source = "8_P.mp4"  # path to video or 0 for webcam
video_name = "8_P_NoT"
check_period = 5000  # ms

cloud = 0
threshold_alarm = 0.82  # threshold for class 1
red = (0, 0, 255)
green = (0, 255, 0)

input_height = 224
input_width = 224

model_name = "resnet50_ht2d"  # options: resnet18, resnet50, mobilenet_v2, efficientnet_b0, swin_t, resnet18_ht2d, resnet50_ht2d
checkpoint_path = "Models/model_checkpoint_Resnet50HT_GWFP_2C_best.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = wht_resnet50()
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print("Numper of parameters in the model:", sum(p.numel() for p in model.parameters()))
model.eval()

def preprocess_bgr(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (input_width, input_height))
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


if not os.path.exists(video_name):
    os.makedirs(video_name)
if not os.path.exists(os.path.join(video_name, "result")):
    os.makedirs(os.path.join(video_name, "result"))



cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {video_source}")

results = np.zeros((45, 2), dtype=np.float32)
frame_idx = 0

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 4.0
out_path = os.path.join(video_name, "result", "annotated.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (1920, 1080))
if not writer.isOpened():
    raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))

    tiles = []
    for m in range(5):
        for n in range(9):
            y0 = m * 180
            x0 = n * 180 + 60
            crop = frame[y0:y0 + 359, x0:x0 + 359, :]
            tiles.append(preprocess_bgr(crop))

    batch = torch.stack(tiles, dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        results[:, :] = probs

    for m in range(7):
        cv2.line(frame, (60, m * 180), (1860, m * 180), green, 2)
    for n in range(11):
        cv2.line(frame, (n * 180 + 60, 0), (n * 180 + 60, 1080), green, 2)

    for m in range(5):
        for n in range(9):
            idx = m * 9 + n
            prob_smoke = results[idx][1]
            if m < cloud:
                cv2.putText(frame, "Off", (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
            else:
                if prob_smoke > threshold_alarm:
                    color = red
                    cv2.rectangle(frame, (n * 180 + 60, m * 180), (n * 180 + 360 + 60, m * 180 + 360), red, 2)
                else:
                    color = green
                text = f"{prob_smoke * 100:.2f}%"
                cv2.putText(frame, text, (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out_path = os.path.join(video_name, "result", f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(out_path, frame)
    print(frame.shape)
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
