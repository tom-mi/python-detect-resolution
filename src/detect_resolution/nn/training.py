import os.path

import torch
from torch.utils.data import DataLoader

from detect_resolution.nn.model import UpscalingDetectorNN, MODEL_PATH
from detect_resolution.nn.training_data import TrainingDataset


def train():
    dataset = TrainingDataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = UpscalingDetectorNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"Loaded existing model from {MODEL_PATH}")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(20):
        for images, scale_factors in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), scale_factors.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()