import argparse
import numpy as np
import nibabel as nib


# The 10 thresholds derived from optimization
THRESHOLDS = [8, 8, 3, 29, 2, 17, 23, 11, 24, 2]


def calculate_score(image_path, template_path, thresholds):
    # Read the atlas template
    template_img = nib.load(template_path)
    reg = template_img.get_fdata()

    reg[np.isnan(reg)] = 0
    reg[np.isinf(reg)] = 0

    # Number of voxels per brain region
    region_size = np.zeros(20)

    for i in range(20):
        region_size[i] = np.sum(reg == (i + 1))

    # Load the lesion mask
    img = nib.load(image_path)
    lesion = img.get_fdata()

    lesion[np.isnan(lesion)] = 0
    lesion[lesion != 0] = 1

    # Calculate the lesion load
    overlap = lesion * reg

    region_ratio = np.zeros(20)

    for i in range(20):
        lesion_voxel = np.sum(overlap == (i + 1))

        if region_size[i] > 0:
            region_ratio[i] = lesion_voxel / region_size[i]
        else:
            region_ratio[i] = 0

    # ASPECTS score
    score = 10

    for i in range(10):
        threshold = thresholds[i] / 100

        if region_ratio[i] > threshold:
            score -= 1

    score = max(score, 0)

    return region_ratio, score


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Ref-AI-ASPECTS score"
    )

    parser.add_argument(
        "--mask",
        required=True,
        help="Input lesion mask (.nii/.nii.gz)"
    )

    parser.add_argument(
        "--atlas",
        required=True,
        help="ASPECTS atlas file (.nii/.nii.gz)"
    )

    args = parser.parse_args()

    region_ratios, final_score = calculate_score(
        args.mask,
        args.atlas,
        THRESHOLDS
    )

    print("=" * 60)
    print("Regional lesion ratios")

    for i in range(20):
        print(
            f"Region {i+1:02d}: {region_ratios[i]:.4f}"
        )

    print("\nThreshold evaluation")

    score = 10

    for i in range(10):

        ratio = region_ratios[i]
        threshold = THRESHOLDS[i] / 100

        if ratio > threshold:
            result = "YES (-1)"
            score -= 1
        else:
            result = "NO"

        print(
            f"Feature {i+1:02d}: "
            f"ratio={ratio:.4f}, "
            f"threshold={threshold:.4f}, "
            f"exceed={result}"
        )

    print("\nFinal Ref-AI-ASPECTS score:", score)
    print("=" * 60)


if __name__ == "__main__":
    main()