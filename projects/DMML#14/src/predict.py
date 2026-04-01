import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2

from torch import nn
from torchvision import models
import argparse
import sys

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

# Model Definition exactly as in train.py
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True) 
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=20, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
        
    def __len__(self):
        return len(self.video_names)
        
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count) if self.count > 0 else 1
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            # Fallback to no face extraction to handle cases without faces efficiently
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        
        while len(frames) < self.count and len(frames) > 0:
            frames.append(frames[-1])
            
        if len(frames) == 0:
            print(f"Warning: Could not extract frames from {video_path}")
            return torch.zeros((self.count, 3, im_size, im_size)).unsqueeze(0)
            
        frames = torch.stack(frames)
        return frames.unsqueeze(0)
        
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def predict(model, img, device):
    img = img.to(device)
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Deepfake from Video")
    parser.add_argument('--video', type=str, required=True, help="Path to the video file")
    parser.add_argument('--model', type=str, default='./checkpoint_v2.pt', help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model} on {device}...")
    
    model = Model(2)
    # Load model properly whether on cpu or gpu
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Processing video: {args.video}...")
    dataset = validation_dataset([args.video], sequence_length=20, transform=train_transforms)
    video_tensor = dataset[0] # Returns shape (1, seq, C, H, W)
    
    with torch.no_grad():
        print("Making prediction...")
        pred, conf = predict(model, video_tensor, device)
        
    label = "REAL" if pred == 1 else "FAKE"
    print(f"\n========================")
    print(f"Result: {label}")
    print(f"Confidence: {conf:.2f}%")
    print(f"========================\n")
