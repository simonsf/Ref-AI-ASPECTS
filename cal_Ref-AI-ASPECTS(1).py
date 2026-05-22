import numpy as np
import nibabel as nib

# ===================== 【Only modify here】 =====================
# The ten optimized region-specific lesion-load thresholds ！！！
THRESHOLDS = [8, 8, 3, 29, 2, 17, 23, 11, 24, 2]  # 10个

# file path
TEMPLATE_PATH = r"D:\\aspect\\atlas\\Reslice_ASPECTS_region_0%.nii"
IMAGE_PATH = r"D:\\programs\\matlab\\vol\\rename_NormROI_All\\STK_CZ1_DWI_00003.nii"
# ===========================================================

def calculate_score(image_path, template_path, thresholds):
    # read the template
    template_img = nib.load(template_path)
    reg = template_img.get_fdata()
    reg[np.isnan(reg)] = 0
    reg[np.isinf(reg)] = 0

    a = np.zeros(20)
    for i in range(20):
        a[i] = np.sum(reg == (i + 1))

    # read the image
    img = nib.load(image_path)
    ma = img.get_fdata()
    ma[np.isnan(ma)] = 0
    ma[ma != 0] = 1

    # calculate the proportion
    cntt = ma * reg
    rate = np.zeros(20)
    for j in range(20):
        x = np.sum(cntt == j + 1)
        rate[j] = x / a[j] if a[j] > 0 else 0

    # ===================== 【Core：automatically÷100】 =====================
    score = 10
    for i in range(10):
        real_th = thresholds[i] / 100   # <-- Here, it must be divided by 100.
        if rate[i] > real_th:
            score -= 1
    # ==============================================================

    score = max(score, 0)
    return rate, score

# ===================== Run =====================
if __name__ == "__main__":
    region_rates, final_score = calculate_score(
        IMAGE_PATH, TEMPLATE_PATH, THRESHOLDS
    )

    print("=" * 60)
    print("📊 The proportion of the lesion load within each of the 20 brain regions：")
    for i in range(20):
        print(f"brain regions{i+1:2d}：{region_rates[i]:.4f}")

    print("\n🎯 The judgment result using the correct threshold (already divided by 100)：")
    score = 10
    for i in range(10):
        rt = region_rates[i]
        th = THRESHOLDS[i] / 100
        over = "✅ exceed → Subtract 1 point" if rt > th else "no"
        if rt > th:
            score -= 1
        print(f"feature{I+1:2d} | proportion={rt:.4f} |  lesion-load threshold={th:.4f} | exceed：{over}")

    print(f"\n🔥 The correct final score = {score}")
    print("=" * 60)