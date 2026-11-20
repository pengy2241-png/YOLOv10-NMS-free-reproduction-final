---
id: S4
title: YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors
year: 2023
venue: CVPR
citation: Wang, Bochkovskiy, and Liao, CVPR 2023, DOI 10.1109/CVPR52729.2023.00721
tags: yolo, yolov7, bag-of-freebies, e-elan, real-time
---
# Core contribution
YOLOv7 focuses on trainable bag-of-freebies, E-ELAN, auxiliary heads and re-parameterization to improve accuracy without increasing inference cost.

# Evidence
The paper reports 56.8% AP among real-time detectors at 30 FPS or higher on V100, and 55.9% AP at 56 FPS for YOLOv7-E6.

# Limitation
The evaluation is primarily COCO-centric and relies on many training-time techniques, increasing reproduction complexity.
