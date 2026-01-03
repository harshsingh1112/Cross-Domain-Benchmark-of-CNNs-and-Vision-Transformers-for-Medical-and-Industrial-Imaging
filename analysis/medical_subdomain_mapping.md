# Medical Domain Analysis

## Sub-Domain Mapping

| Sub-Domain | Status | Key Characteristics / Challenges | Literature Findings (Inferred) |
| :--- | :--- | :--- | :--- |
| **1. Histopathology (PathMNIST)** | **Evaluated** | Multi-class tissue classification; texture-heavy. | [Insert Experimental Results Here] |
| 2. Chest X-ray | Inferred | Grayscale, low contrast, subtle features. | CNNs typically strong; ViTs require large data. |
| 3. Brain MRI | Inferred | 3D volumetric data often treated as 2D slices. | ResNet widely used baseline. |
| 4. Skin Lesion | Inferred | Color/texture crucial; high variability. | EfficientNet often performs well due to scaling. |
| 5. Retinal Fundus | Inferred | Fine vessel details; high resolution needed. | ViTs showing promise for global context. |
| 6. OCT Imaging | Inferred | Cross-sectional noise; strictly structural. | CNNs dominate for noise robustness. |
| 7. Ultrasound | Inferred | High speckle noise; low resolution. | Simple CNNs often sufficient. |
| 8. Mammography | Inferred | high-res; detection of micro-calcifications. | Specialized high-res CNNs needed. |
| 9. Bone Fracture | Inferred | Geometric edge detection. | CNNs (edges) > ViTs (texture). |
| 10. CT/MRI Organ | Inferred | Segmentation/Classification mix. | 3D-CNNs or U-Nets standard. |

## Experimental Findings (PathMNIST)
*Replace this with summary of your results.*

## Discussion
*Contextualize why one model performed better based on the nature of histopathology images (texture vs object shape).*
