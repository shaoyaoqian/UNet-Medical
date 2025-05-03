from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file = h5py.File(file_path, 'r')  # 注意多进程下需特殊处理
        self.f = self.file['f']
        self.u = self.file['u']
        self.transform = transform

    def __len__(self):
        assert self.f.shape == self.u.shape, "两个数据集形状不一致"
        return len(self.f)

    def __getitem__(self, idx):
        u, f = torch.from_numpy(self.f[idx]), torch.from_numpy(self.u[idx])
        u, f = u.unsqueeze(0), f.unsqueeze(0)
        return u, f

# 使用DataLoader批量读取
# dataset = HDF5Dataset('from_memory.h5')

class NpyDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file = h5py.File(file_path, 'r')  # 注意多进程下需特殊处理
        self.f = self.file['f']
        self.u = self.file['u']
        self.transform = transform

    def __len__(self):
        assert self.f.shape == self.u.shape, "两个数据集形状不一致"
        return len(self.f)

    def __getitem__(self, idx):
        u, f = torch.from_numpy(self.f[idx]), torch.from_numpy(self.u[idx])
        u, f = u.unsqueeze(0), f.unsqueeze(0)
        return u, f
    


class NpyDataset(Dataset):
    def __init__(self, file, transform=None):
        # 使用np.load直接加载.npy文件
        self.data = np.load(file)
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        # 获取对应索引的数据
        f = self.data[0][idx].astype(np.float32)  # 转换为float32
        u = self.data[1][idx].astype(np.float32)
        
        # 计算f的平均值
        f_mean = f.mean()

        # 归一化处理
        f_normalized = (f / f_mean) - 1
        u_normalized = (u / f_mean) - 1

        # 添加通道维度 (C=1)
        # f_tensor = torch.from_numpy(f_normalized).unsqueeze(0)  # 形状 (1, H, W)
        # u_tensor = torch.from_numpy(u_normalized).unsqueeze(0)
        
        f_tensor = torch.from_numpy(f).unsqueeze(0)  # 形状 (1, H, W)
        u_tensor = torch.from_numpy(u).unsqueeze(0)
        
        # if self.transform:
        #     f_tensor = self.transform(f_tensor)
        #     u_tensor = self.transform(u_tensor)
            
        return f_tensor, u_tensor # 注意顺序与原始代码保持一致
