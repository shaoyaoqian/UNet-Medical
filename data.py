from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import h5py

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