---
id: S3
title: YOLOv10: Real-Time End-to-End Object Detection
year: 2024
venue: NeurIPS
citation: Wang et al., NeurIPS 2024, DOI 10.52202/079017-3429
tags: yolo, yolov10, nms-free, end-to-end, consistent-dual-assignment, real-time
---
# Core contribution
YOLOv10 identifies NMS as a post-processing bottleneck for end-to-end deployment. It introduces consistent dual assignments: a one-to-many branch provides dense supervision, while a one-to-one branch is trained with a consistent matching metric and used directly at inference without NMS.

# Evidence
The paper reports that YOLOv10-S is about 1.8 times faster than RT-DETR-R18 at similar AP on COCO while using about 2.8 times fewer parameters and FLOPs. It also reports that YOLOv10-B has 46% lower latency and 25% fewer parameters than YOLOv9-C at the same performance.

# Limitation
The claims are evaluated mainly on COCO and specific hardware. Dense-scene behavior, calibration, and cross-device latency require independent validation.
