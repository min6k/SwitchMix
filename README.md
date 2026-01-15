# SwitchMix: Switching Dual-Level Mix Teacher for Domain Adaptive Thermal Object Detection

This repository provides the official implementation of SwitchMix, a novel framework for unsupervised domain adaptation (UDA) from RGB to thermal infrared (TIR) object detection.

Overview

Adapting object detectors from RGB to thermal images is challenging due to the severe cross-modality gap and unreliable pseudo-labels in the target domain.
SwitchMix addresses these issues by introducing a dual-level mix-teacher framework combined with a switching supervision strategy, enabling stable and effective RGB-to-thermal adaptation without adversarial training.

Key Ideas

Dual-level domain mixing

Image-level mixing: aligns local spectral and texture cues via block-wise RGB–thermal blending

Sample-level mixing: aligns global style distributions using Fourier Domain Adaptation (FDA)

Switching dual-teacher supervision

Periodically alternates between complementary teachers to reduce bias and stabilize pseudo-labels

Progressive RGB-to-thermal adaptation

Gradually shifts supervision from RGB-centric to thermal-centric representations

Results

SwitchMix achieves state-of-the-art performance on FLIR and KAIST datasets for RGB-to-thermal object detection, demonstrating improved accuracy and robustness under large modality gaps.

Paper

SwitchMix: Switching Dual-Level Mix Teacher for Domain Adaptive Thermal Object Detection
ICEIC 2026
