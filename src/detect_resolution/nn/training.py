import os.path

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from detect_resolution.nn.model import UpscalingDetectorNN, MODEL_PATH
from detect_resolution.nn.training_data import TrainingDataset


def train():
    dataset = TrainingDataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = UpscalingDetectorNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    metrics = []
    for epoch in range(50):
        for images, scale_factors in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), scale_factors.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        metrics.append((epoch, loss.item()))
    torch.save(model.state_dict(), MODEL_PATH)

    plt.figure()
    plt.plot([m[0] for m in metrics], [m[1] for m in metrics])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()


if __name__ == "__main__":
    train()
