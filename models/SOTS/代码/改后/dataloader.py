import torch
import os
from natsort import natsorted
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.transforms import Resize


class MyData(Dataset):
    def __init__(self, path, image_size=(512, 512)):
        self.path = path
        self.image_size = image_size
        self.hazy_dir = os.path.join(path, 'Hazy')
        self.gt_dir = os.path.join(path, 'GT')

        # 1. 收集所有GT文件（兼容png/PNG），并提取基础编号（如1.png -> 1）
        self.gt_files = {}
        for gt_name in os.listdir(self.gt_dir):
            # 统一转为小写，兼容PNG/png后缀
            gt_name_lower = gt_name.lower()
            if gt_name_lower.endswith('.png'):
                # 提取GT文件的基础编号（如"1.png" -> "1"）
                gt_base = gt_name_lower.replace('.png', '')
                # 存储原始文件名（保留大小写），方便后续读取
                self.gt_files[gt_base] = os.path.join(self.gt_dir, gt_name)

        # 2. 收集所有Hazy文件，按基础编号分组（如1_xxx.png归到"1"组）
        self.hazy_groups = {}
        for hazy_name in os.listdir(self.hazy_dir):
            hazy_name_lower = hazy_name.lower()
            if hazy_name_lower.endswith('.png'):
                # 提取Hazy文件的基础编号（如"1_1_0.90179.png" -> "1"）
                hazy_base = hazy_name_lower.split('_')[0]
                hazy_path = os.path.join(self.hazy_dir, hazy_name)
                # 按基础编号分组存储
                if hazy_base not in self.hazy_groups:
                    self.hazy_groups[hazy_base] = []
                self.hazy_groups[hazy_base].append(hazy_path)

        # 3. 整理最终的数据集列表：(hazy_path, gt_path) 配对
        self.dataset = []
        for base_num in self.hazy_groups:
            if base_num in self.gt_files:  # 确保有对应的GT文件
                gt_path = self.gt_files[base_num]
                # 对每组Hazy文件自然排序（保证1_1、1_2...1_10的顺序）
                hazy_paths = natsorted(self.hazy_groups[base_num])
                for hazy_path in hazy_paths:
                    self.dataset.append((hazy_path, gt_path))

        # 若没有匹配的文件，抛出提示
        if len(self.dataset) == 0:
            raise ValueError("未找到匹配的Hazy和GT文件，请检查文件命名和目录结构！")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取配对的雾图和GT图路径
        hazy_path, gt_path = self.dataset[idx]

        # 读取图像（转为RGB，避免单通道问题）
        try:
            hazy_img = Image.open(hazy_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"读取图像失败：{hazy_path} / {gt_path}，错误：{e}")

        # 调整尺寸
        resize = transforms.Resize(self.image_size)
        hazy_img = resize(hazy_img)
        gt_img = resize(gt_img)

        # 转为tensor（无需归一化，若需要可取消注释）
        hazy_tensor = transforms.functional.to_tensor(hazy_img)
        gt_tensor = transforms.functional.to_tensor(gt_img)

        # 可选：归一化（如需启用，取消以下两行注释）
        # norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # hazy_tensor, gt_tensor = norm(hazy_tensor), norm(gt_tensor)

        return hazy_tensor, gt_tensor


class MyData_Test(Dataset):
    def __init__(self, path, resize_dimen=(1024, 1024)):
        self.path = path
        self.resize_dimen = resize_dimen
        self.hazy_dir = os.path.join(path, 'Hazy')
        self.gt_dir = os.path.join(path, 'GT')

        # 1. 初始化Resize变换（与原代码逻辑一致）
        self.resize = Resize(self.resize_dimen)

        # 2. 收集GT文件，按基础编号映射（如1.png -> "1"）
        self.gt_map = {}
        for gt_name in os.listdir(self.gt_dir):
            # 兼容png/PNG大小写后缀
            gt_name_lower = gt_name.lower()
            if gt_name_lower.endswith('.png'):
                # 提取GT的基础编号（1.png -> "1"）
                gt_base = gt_name_lower.replace('.png', '')
                self.gt_map[gt_base] = os.path.join(self.gt_dir, gt_name)

        # 3. 收集Hazy文件，按基础编号分组（1_1.png -> "1"，归入"1"组）
        self.hazy_groups = {}
        for hazy_name in os.listdir(self.hazy_dir):
            hazy_name_lower = hazy_name.lower()
            if hazy_name_lower.endswith('.png'):
                # 提取Hazy的基础编号（1_1.png -> "1"）
                hazy_base = hazy_name_lower.split('_')[0]
                hazy_path = os.path.join(self.hazy_dir, hazy_name)
                if hazy_base not in self.hazy_groups:
                    self.hazy_groups[hazy_base] = []
                self.hazy_groups[hazy_base].append(hazy_path)

        # 4. 构建最终的测试数据集列表：(hazy_path, gt_path) 配对
        self.dataset = []
        for base_num in self.hazy_groups:
            if base_num in self.gt_map:  # 确保有对应的GT文件
                gt_path = self.gt_map[base_num]
                # 对同编号的Hazy文件自然排序（保证1_1、1_2...1_10的顺序）
                sorted_hazy_paths = natsorted(self.hazy_groups[base_num])
                for hazy_path in sorted_hazy_paths:
                    self.dataset.append((hazy_path, gt_path))

        # 空数据校验，避免测试时无样本
        if len(self.dataset) == 0:
            raise ValueError("未匹配到任何Hazy和GT文件，请检查文件命名或目录结构！")

    def __len__(self):
        # 返回匹配后的总样本数（所有Hazy文件数）
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取配对的雾图和GT图路径
        hazy_path, gt_path = self.dataset[idx]

        # 读取图像（转为RGB，避免单通道/四通道问题）
        try:
            real = Image.open(hazy_path).convert('RGB')
            condition = Image.open(gt_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"读取图像失败：{hazy_path} / {gt_path}，错误：{e}")

        # 调整尺寸（复用初始化的Resize变换）
        real = self.resize(real)
        condition = self.resize(condition)

        # 转为tensor（测试阶段无需归一化，保持原逻辑）
        real = transforms.functional.to_tensor(real)
        condition = transforms.functional.to_tensor(condition)

        return real, condition


class MyData_Test_Single(Dataset):
    def __init__(self, path, resize_dimen=(1024, 1024)):
        # Initialize the filename list first
        self.filename_original = sorted(os.listdir(os.path.join(path, 'Hazy')), key=len)
        self.filename_original = natsorted(self.filename_original)

        # Update paths
        self.filename_original = [os.path.join(path, 'Hazy', filename)
                                  for filename in self.filename_original]

        # Initialize the resize transform
        self.resize = Resize(resize_dimen)

    def __len__(self):
        return len(self.filename_original)

    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]

        # Open and process the image
        real = Image.open(filename_o).convert('RGB')
        real = self.resize(real)
        real = transforms.functional.to_tensor(real)

        return real, real  # Return twice to match the expected format in test.py