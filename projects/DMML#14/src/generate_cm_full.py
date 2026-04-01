import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import glob
import pandas as pd
from torch import nn
from torchvision import models
import sys
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

video_files = glob.glob('./dataset/**/*.mp4', recursive=True)
if not video_files:
    video_files = glob.glob('./dataset/*.mp4')

random.seed(42)
random.shuffle(video_files)
valid_video_files = []
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames >= 10:
        valid_video_files.append(video_file)
video_files = valid_video_files

header_list = ["file","label"]
labels = pd.read_csv('./labels/Gobal_metadata.csv', names=header_list)
# Using full validation set ~1290 videos
valid_videos = video_files[int(0.8*len(video_files)):]
print(f"Total Validation size: {len(valid_videos)}")

class video_dataset(Dataset):
    def __init__(self,video_names,labels,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count) if self.count > 0 else 1
        temp_video = video_path.split('/')[-1]
        
        matching_rows = self.labels.loc[self.labels["file"] == temp_video]
        if len(matching_rows) == 0:
            label = 1
        else:
            label_text = self.labels.iloc[matching_rows.index.values[0], 1]
            if label_text == 'FAKE':
                label = 0
            else:
                label = 1
                
        for i,frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        
        while len(frames) < self.count and len(frames) > 0:
            frames.append(frames[-1])
            
        if len(frames) == 0:
            frames = [torch.zeros((3, im_size, im_size)) for _ in range(self.count)]
            
        frames = torch.stack(frames)
        return frames,label
        
    def frame_extract(self,path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True) 
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))

print("Loading trained model weights for full confusion matrix evaluation...")
model = Model(2)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('./checkpoint_v2.pt'))
    model = model.cuda()
else:
    model.load_state_dict(torch.load('./checkpoint_v2.pt', map_location=torch.device('cpu')))

model.eval()
frames_to_test = [10, 20, 40, 60, 80, 100]

for f in frames_to_test:
    val_data = video_dataset(valid_videos, labels, sequence_length=f, transform=train_transforms)
    bs = 2 if f <= 20 else 1
    valid_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=2)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            if torch.cuda.is_available():
                targets = targets.cuda()
                inputs = inputs.cuda()
            _,outputs = model(inputs)
            
            _, pred = outputs.topk(1, 1, True)
            pred = pred.t()
            
            true_labels.extend(targets.cpu().numpy())
            pred_labels.extend(pred.view(-1).cpu().numpy())
            
            sys.stdout.write(f"\r[Frames: {f}] Evaluating for full Confusion Matrix... [{i+1}/{len(valid_loader)}]")
            sys.stdout.flush()

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8,6))
    # 0 is Fake, 1 is Real
    sn.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Deepfake Model Full Confusion Matrix ({f} Frames)')
    plt.tight_layout()

    # Save to current folder explicitly denoting it's the full version
    dest_path = f'./confusion_matrix_full_{f}_frames.png'
    plt.savefig(dest_path)
    plt.close()
    sys.stdout.write(f"\nFull Confusion matrix for {f} frames successfully saved as {dest_path}\n")
    sys.stdout.flush()
