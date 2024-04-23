import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def rotate_with_grid_sample(image, angle):
    # 将PIL图像转换为torch张量
    transform = T.Compose([T.ToTensor()])
    tensor = transform(image).unsqueeze(0)  # 添加batch维度

    # 创建旋转网格
    theta = torch.tensor([
        [np.cos(np.radians(angle)), np.sin(-np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
    ], dtype=torch.float)

    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), tensor.size(), align_corners=False)
    
    # 使用grid_sample进行采样
    rotated_tensor = torch.nn.functional.grid_sample(tensor, grid, align_corners=False)
    
    # 将结果张量转换回PIL图像
    rotated_image = T.ToPILImage()(rotated_tensor.squeeze(0))
    return rotated_image


def visualize_image_operations(image_path):
    # 加载图像
    img = Image.open(image_path)
    
    # 裁剪图像（例如，裁剪中心区域）
    width, height = img.size
    new_width, new_height = width // 2, height // 2
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    cropped_img = img.crop((left, top, right, bottom))
    
    # 调整大小（例如，将图像缩小一半）
    resized_img = img.resize((width // 2, height // 2))
    
    # 网格采样：这里我们使用旋转作为示例
    angle = 45  # 旋转角度
    rotated_image = rotate_with_grid_sample(img, angle)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cropped_img)
    plt.title('Cropped Image')
    
    plt.subplot(2, 2, 3)
    plt.imshow(resized_img)
    plt.title('Resized Image')

    plt.subplot(2, 2, 4)
    plt.imshow(rotated_image)
    plt.title(f'Rotated Image by {angle} Degrees')
    plt.savefig("./person.png")

    plt.tight_layout()
    plt.show()


# 替换以下路径为你的图片路径
image_path = "/mnt/share_disk/bruce_cui/Pytorch_learning/Deploy/face_ort_3.png"
visualize_image_operations(image_path)
