---
id: S1
title: You Only Look Once: Unified, Real-Time Object Detection
year: 2016
venue: CVPR
citation: Redmon et al., CVPR 2016, DOI 10.1109/CVPR.2016.91
tags: yolo, real-time, closed-set, single-stage, regression
---
# Core contribution
YOLO frames object detection as a single regression problem. One neural network predicts bounding boxes and class probabilities directly from a full image in one evaluation.

# Evidence
The base model runs at 45 FPS; Fast YOLO runs at 155 FPS. The authors report stronger generalization to artwork than several prior detection methods.

# Limitation
The paper explicitly notes more localization errors and weaker performance on small or overlapping objects. The detector is closed-set: its output classes are fixed by the training annotation set.
