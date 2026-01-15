# SwitchMix:  **Switching Dual-Level Mix Teacher for Domain Adaptive Thermal Object Detection**


## 🔍 Overview
Adapting object detectors from **RGB to thermal infrared (TIR)** images is challenging due to the **severe cross-modality gap** and **unreliable pseudo-labels** in the target domain.

**SwitchMix** addresses this problem by introducing a  **dual-level mix-teacher framework** with a **switching supervision strategy**, enabling **stable and effective RGB-to-thermal adaptation without adversarial training**.



## 💡 Key Contributions
- **Dual-level domain mixing**
  - **Image-level mixing**: aligns *local spectral and texture cues* via block-wise RGB–thermal blending
  - **Sample-level mixing**: aligns *global style distributions* using **Fourier Domain Adaptation (FDA)**

- **Switching dual-teacher supervision**
  - Periodically alternates between **complementary teachers**
  - Reduces **pseudo-label noise** and **confirmation bias**

- **Progressive RGB → Thermal adaptation**
  - Gradually shifts supervision from **RGB-centric** to **thermal-centric** representations


## 📊 Results
SwitchMix achieves **state-of-the-art performance** on:
- **FLIR Dataset**
  <img width="1654" height="366" alt="image" src="https://github.com/user-attachments/assets/f2f2140f-cc5d-4e01-85b2-18c93cac1fdc" />

- **KAIST Dataset**
  <img width="1678" height="351" alt="image" src="https://github.com/user-attachments/assets/4a666cea-bb9a-457c-ac98-8d049a379165" />
  
- **Performance FLIR & KAIST Dataset**
  
  <img width="530" height="276" alt="image" src="https://github.com/user-attachments/assets/3dc736fc-34f6-4154-9de9-f90d9f4b15af" />
  <img width="280" height="276" alt="image" src="https://github.com/user-attachments/assets/27731f7d-ffe9-418e-8593-090f7a2e8b80" />



demonstrating improved **accuracy** and **robustness** under large modality gaps.

## 📄 Paper
**SwitchMix: Switching Dual-Level Mix Teacher for Domain Adaptive Thermal Object Detection**  
*ICEIC 2026* accepted.
