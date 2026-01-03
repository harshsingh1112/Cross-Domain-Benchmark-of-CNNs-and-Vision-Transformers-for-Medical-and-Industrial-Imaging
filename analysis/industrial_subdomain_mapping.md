# Industrial Domain Analysis

## Sub-Domain Mapping

| Sub-Domain | Status | Key Characteristics / Challenges | Literature Findings (Inferred) |
| :--- | :--- | :--- | :--- |
| **1. Remote Sensing (EuroSAT)** | **Evaluated** | Multispectral (used RGB), texture & object mix. | [Insert Experimental Results Here] |
| 2. Aerial UAV | Inferred | Variable scale/orientation; top-down view. | Rotation-invariant CNNs or ViTs useful. |
| 3. Astronomy | Inferred | Sparse signals (stars) or diffuse (galaxies). | Specialized architectures often needed. |
| 4. Climate/Weather | Inferred | Fluid patterns; temporal dynamics often key. | ResNets standard for static frames. |
| 5. Energy Infra | Inferred | Surveillance/monitoring; anomaly detection. | EfficientNet good for edge deployment. |
| 6. Materials Science | Inferred | Microstructure texture; similar to histology. | Texture-biased models (CNNs) strong. |
| 7. Aerospace Inspection | Inferred | Defect detection (cracks); high precision. | High-res inputs required. |
| 8. Geophysical | Inferred | Seismic lines; signal processing flavor. | 1D/2D CNNs. |
| 9. Optical Microscopy | Inferred | Non-biological; crystal structures/defects. | Similar to materials science. |
| 10. Fluid Dynamics | Inferred | Flow fields; colorized visualisations. | CNNs for pattern recognition. |

## Experimental Findings (EuroSAT)
*Replace this with summary of your results.*

## Discussion
*Contextualize results: ViTs often excel in remote sensing due to global context requirements, whereas CNNs are robust baselines.*
