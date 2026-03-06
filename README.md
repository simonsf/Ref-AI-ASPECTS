# Ref-AI-ASPECTS

Based on the scoring practices of experts, we have optimized the regions of the ASPECTS scoring template for ischemic stroke. Through experiments examining the correlation between our scoring method and prognostic predictions, we have defined a superior scoring approach, as shown in Equation 1:

$
\frac{\text{LesionVolume}_{\text{newTemplate}}}{\text{RegionVolume}_{\text{oldTemplate}}} \times 100\% > \text{threshold}
$

Here, $\text{LesionVolume}_{\text{newTemplate}}$ refers to the ischemic volume of each region as defined by the new template, and $\text{RegionVolume}_{\text{oldTemplate}}$ denotes the volume of the corresponding region in the original classic template. When the percentage for a given region exceeds a predefined threshold, that region is considered ischemic, and one point is deducted from the patient’s total score. The thresholds for each region are provided in the `threshold.py` file.
