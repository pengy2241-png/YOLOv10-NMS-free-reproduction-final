---
id: S5
title: YOLO-World: Real-Time Open-Vocabulary Object Detection
year: 2024
venue: CVPR
citation: Cheng et al., CVPR 2024, DOI 10.1109/CVPR52733.2024.01599
tags: yolo, open-vocabulary, closed-set, vision-language, repvl-pan, real-time
---
# Core contribution
YOLO-World addresses the closed-set limitation of YOLO by adding vision-language modeling. It proposes a Re-parameterizable Vision-Language Path Aggregation Network (RepVL-PAN) and a region-text contrastive loss to connect visual and linguistic information.

# Evidence
The paper reports 35.4 AP at 52 FPS on V100 on LVIS in a zero-shot setting, and demonstrates downstream fine-tuning for detection and open-vocabulary instance segmentation.

# Limitation
The method depends on a text encoder, large-scale vision-language pre-training, vocabulary choice and prompt design. Zero-shot results do not replace domain-specific validation in specialized fields.
