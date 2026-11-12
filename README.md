# TRACE-FrED: low-cost thermal imaging of fiber cooling in the FrED

Data, analysis code and a web-based digital twin for the paper *Low-cost thermal imaging of fiber cooling in the FrED: a case study of empirical and physics-based models in a learning factory* (NAMRC 55).

**TRACE-FrED** = Thermal Reconstruction of Axial profiles from Cheap sensors with Embedded physics.

## What is here
| Folder | Content |
|---|---|
| `twin/` | Single-file web app (`index.html`): semi-physical digital twin. Open it in a browser (it loads fonts and MathJax from the internet). |
| `code/` | Python scripts that reproduce the paper's results. |
| `data/` | 78 thermal images (3 per condition, 26 conditions of a 3×3×3 design), calibration file and fiber diameters. |

## Reproduce the analysis
```bash
pip install -r code/requirements.txt
bash code/run_all.sh        # outputs go to out/
```
Steps: image processing (`01`), characterization and ANOVA (`02`), empirical, semi-physical, physics-based and hybrid models with leave-one-condition-out validation (`03`), PINN (`03b`, needs PyTorch), metrics and figures (`04`, `05`). The physics-based model and the PINN take about 20 to 30 minutes each on two cores.

## Notes
- The material (glue-gun sticks) has no datasheet; ρ = 1200 kg/m³ and c_p = 1300 J/(kg K) are assumptions that only affect the heat-transfer coefficient, not the empirical or semi-physical predictions.
- Temperatures are apparent: the fiber is narrower than one camera pixel.

## Licenses
Code: MIT (see `LICENSE`). Data: CC BY 4.0.

## Citation
[Add the paper reference and the Zenodo DOI when available.]
