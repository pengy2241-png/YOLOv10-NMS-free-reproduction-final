---
id: S2
title: YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
year: 2024
venue: ECCV
citation: Wang, Yeh, and Liao, ECCV 2024, DOI 10.1007/978-3-031-72751-1_1
tags: yolo, yolov9, gradient, information-bottleneck, gelan, real-time
---
# Core contribution
YOLOv9 targets information loss during layer-by-layer feature extraction and spatial transformation. It proposes Programmable Gradient Information (PGI) to provide reliable gradient information for target objectives, and Generalized Efficient Layer Aggregation Network (GELAN) as a lightweight architecture based on gradient-path planning.

# Evidence
The paper reports improved parameter utilization with conventional convolution operators and better information retention for train-from-scratch models. Experiments are centered on MS COCO object detection.

# Limitation
The main evidence is COCO detection. Cross-task generality and behavior on specialized domains remain open questions.
