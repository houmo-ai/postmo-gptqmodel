from torch.utils.data import Dataset
import random
import torch
import re
import os
import gc


class R1Dataset(Dataset):
    def __init__(self, data_dir, nsamples, shuffle=False):
        """
        Args:
            data_dir (str): .pt 文件存储的目录路径
        """
        self.data_dir = data_dir  # 存放 .pt 文件的目录
        self.file_list = self._load_file_list(nsamples)  # 加载所有 .pt 文件
        self.shuffle = shuffle

    def _load_file_list(self, nsamples: int):
        """
        遍历数据目录，找到所有 .pt 文件并返回文件名列表
        """
        # file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        file_list = [
            f for f in os.listdir(self.data_dir)
            if f.endswith('.pt') and re.match(r'sample_(\d{1,2})_', f) and 0 <= int(re.match(r'sample_(\d{1,2})_', f).group(1)) <= nsamples - 1
        ]
        file_list.sort()
        return file_list

    def _shuffle_data(self, sample):
        """Shuffle num_b and tokens dimensions."""
        perm1 = torch.randperm(self.current_data.shape[0])
        sample = sample[perm1, :]
        return sample

    def __len__(self):
        # 返回样本总数（即 .pt 文件的个数）
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        根据索引 idx 加载对应的 .pt 文件。
        Args:
            idx (int): 样本的索引
        Returns:
            Tensor: 返回从 .pt 文件加载的张量
        """
        # 获取文件路径
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample = torch.load(file_path, map_location='cpu', weights_only=True)  # 加载 .pt 文件
        if self.shuffle:
            sample = self._shuffle_data(sample)
        return sample


class R2Dataset(Dataset):
    def __init__(self, data_path, nsamples, dev):
        self.data = torch.load(data_path, map_location=dev)[:nsamples, :, :]
        # print(self.data.shape)

        # Shuffle along the 0th and 1st dimensions
        self.data = self.data[torch.randperm(self.data.size(0)), :, :]
        self.data = self.data[:, torch.randperm(self.data.size(1)), :]

        # self.device = dev
        # # Move the combined tensor to GPU
        # self.data = self.data.to(self.device, non_blocking=True)

    def __len__(self):
        # The length of the dataset is the number of [2048, 4096] samples
        return self.data.size(0)

    def __getitem__(self, idx):
        # Directly return the preprocessed sample
        return self.data[idx]  # Each item is [2048, 4096]


class R4Dataset(Dataset):
    def __init__(self, data_path, nsamples, dev):
        self.data = torch.load(data_path, map_location=dev)[:nsamples, :, :]
        self.data = self.data[torch.randperm(self.data.size(0)), :, :]
        self.data = self.data[:, torch.randperm(self.data.size(1)), :]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class R1LayerDataset(Dataset):
    def __init__(self, data_dir, layer_id, nsamples, modules=None, shuffle=False):
        """
        Args:
            data_dir (str): directory with per-layer .pt files.
            layer_id (int): layer index to filter.
            nsamples (int): max sample index to include.
            modules (list[str]): optional module name filter.
        """
        self.data_dir = data_dir
        self.layer_id = int(layer_id)
        self.nsamples = int(nsamples)
        self.modules = set(modules) if modules is not None else None
        self.shuffle = shuffle
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        pattern = re.compile(r'sample_(\d+)_layer_(\d+)_([A-Za-z0-9_]+)\.pt$')
        file_list = []
        for name in os.listdir(self.data_dir):
            if not name.endswith('.pt'):
                continue
            match = pattern.match(name)
            if match is None:
                continue
            sample_id = int(match.group(1))
            layer_id = int(match.group(2))
            module_name = match.group(3)
            if sample_id >= self.nsamples or layer_id != self.layer_id:
                continue
            if self.modules is not None and module_name not in self.modules:
                continue
            file_list.append(name)
        file_list.sort()
        return file_list

    def __len__(self):
        return len(self.file_list)

    def _shuffle_data(self, sample):
        perm1 = torch.randperm(sample.shape[0])
        return sample[perm1, :]

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample = torch.load(file_path, map_location='cpu', weights_only=True)
        if self.shuffle:
            sample = self._shuffle_data(sample)
        return sample
