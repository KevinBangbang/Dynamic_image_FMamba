# Name: Bangcheng Wang
# time:
# train_model.py

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, num_epochs, device):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CompositeLoss().to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, inputs)  # 假设输入时空特征相同，调整这里
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs, val_inputs)  # 同上
                val_loss += criterion(val_outputs, val_targets).item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

    return model
