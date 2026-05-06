import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# 配置路径
input_validation_path = Path(r"/\train_input\train_Gen")
gt_dir = input_validation_path / "GT"
hazy_dir = input_validation_path / "Hazy"

# 输出路径
output_patch_path = Path(r"/\train_input\Train")
output_gt_patch = output_patch_path / "GT"
output_hazy_patch = output_patch_path / "Hazy"

for dir_path in [output_patch_path, output_gt_patch, output_hazy_patch]:
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

# 配置参数
patch_size = 768  # 补丁尺寸
stride = 512  # 滑动窗口步长

img_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor [C, H, W]，值范围0-1
])


def extract_patches(image_path, output_dir, prefix, patch_size, stride):
    """
    从单张图像提取滑动窗口补丁（使用PIL/torchvision读取和处理）
    :param image_path: 输入图像路径
    :param output_dir: 补丁输出目录
    :param prefix: 补丁文件名前缀（区分不同原图）
    :param patch_size: 补丁尺寸
    :param stride: 滑动步长
    """
    try:
        img = Image.open(str(image_path)).convert('RGB')
        img_tensor = img_transform(img)  # [3, H, W]
        C, H, W = img_tensor.shape

        patch_idx = 0

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # [C, patch_size, patch_size]
                patch_tensor = img_tensor[:, y:y + patch_size, x:x + patch_size]

                patch_pil = transforms.ToPILImage()(patch_tensor)

                patch_filename = f"{prefix}_patch_{patch_idx:03d}.jpg"
                patch_save_path = output_dir / patch_filename

                patch_pil.save(str(patch_save_path), quality=95)
                patch_idx += 1

        print(f"已从 {image_path.name} 提取 {patch_idx} 个补丁")

    except Exception as e:
        print(f"错误：处理图像 {image_path} 时失败 - {str(e)}")
        return


# 获取所有GT和雾天图像对（按文件名匹配）
gt_files = sorted([f for f in gt_dir.glob("*.jpg") if "GT" in f.name])
hazy_files = sorted([f for f in hazy_dir.glob("*.jpg") if "hazy" in f.name])

assert len(gt_files) == len(hazy_files), "GT图像和雾天图像数量不匹配！"

print(f"开始提取补丁，共找到 {len(gt_files)} 组图像对...")
for gt_file, hazy_file in tqdm(zip(gt_files, hazy_files), total=len(gt_files)):

    prefix = gt_file.name.replace("_GT.jpg", "")

    hazy_prefix = hazy_file.name.replace("_hazy.jpg", "")
    assert prefix == hazy_prefix, f"文件名不匹配：{gt_file.name} 和 {hazy_file.name}"

    extract_patches(gt_file, output_gt_patch, prefix, patch_size, stride)

    extract_patches(hazy_file, output_hazy_patch, prefix, patch_size, stride)

gt_patch_count = len(list(output_gt_patch.glob('*.jpg')))
hazy_patch_count = len(list(output_hazy_patch.glob('*.jpg')))

print(f"\n补丁提取完成！")
print(f"输出目录：{output_patch_path}")
print(f"GT补丁总数：{gt_patch_count}")
print(f"雾天补丁总数：{hazy_patch_count}")