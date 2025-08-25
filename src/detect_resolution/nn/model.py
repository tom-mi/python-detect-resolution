import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'upscaling_detector.pth')
INPUT_SIZE = 128


class UpscalingDetectorNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 32, 32]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model() -> UpscalingDetectorNN:
    model = UpscalingDetectorNN()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model
