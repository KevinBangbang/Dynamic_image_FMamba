# Name: Bangcheng Wang
# time:
# main.py

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data
from network_architecture import FusionMamba
from train_model import train_model
from evaluate_model import evaluate_model

def load_data(data_dir):
    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
    images = preprocess_data(image_paths)
    images = torch.tensor(images, dtype=torch.float32)
    dataset = TensorDataset(images, images)  # 假设目标是输入本身，调整这里
    return DataLoader(dataset, batch_size=4, shuffle=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader = load_data('path_to_train_data')
    val_loader = load_data('path_to_val_data')

    # 定义模型
    model = FusionMamba()

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    # 评估模型
    avg_psnr, avg_ssim = evaluate_model(trained_model, val_loader, device)
    print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")

if __name__ == "__main__":
    main()
