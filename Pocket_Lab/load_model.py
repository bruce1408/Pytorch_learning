import torch

qat_model = torch.load("/mnt/share_disk/bruce_cui/Yolop_chip/runs/BddDataset/_2023-05-14-13-37/epoch-73.pth", map_location="cpu")
print(qat_model['optimizer']["param_groups"][0]['lr'])