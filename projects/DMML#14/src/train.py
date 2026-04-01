import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
from torch import nn
from torchvision import models
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
import seaborn as sn

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

# 1. Load video files from local directory
video_files = glob.glob('./dataset/**/*.mp4', recursive=True)
if not video_files:
    video_files = glob.glob('./dataset/*.mp4')
print("Total number of videos found:", len(video_files))

random.seed(42)
random.shuffle(video_files)

# Filter videos that have fewer than 100 frames (as per original code)
valid_video_files = []
frame_count = []
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames >= 10: 
        valid_video_files.append(video_file)
        frame_count.append(total_frames)

video_files = valid_video_files
print("Total valid videos after frame check:", len(video_files))
if len(video_files) == 0:
    print("No valid videos found. Exiting.")
    sys.exit(0)

print('Average frames per video:', np.mean(frame_count))

# 2. Dataset Definition
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
        a = int(100/self.count)
        first_frame = np.random.randint(0,a) if a > 0 else 0
        temp_video = video_path.split('/')[-1]
        
        matching_rows = self.labels.loc[self.labels["file"] == temp_video]
        if len(matching_rows) == 0:
            print(f"Warning: {temp_video} not found in labels. Defaulting to REAL.")
            label = 1
        else:
            label_text = self.labels.iloc[matching_rows.index.values[0], 1]
            if label_text == 'FAKE':
                label = 0
            else:
                label = 1 # REAL
                
        for i,frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        
        while len(frames) < self.count:
            frames.append(frames[-1])
            
        frames = torch.stack(frames)
        return frames,label
        
    def frame_extract(self,path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# 3. Load labels from local file
header_list = ["file","label"]
labels = pd.read_csv('./labels/Gobal_metadata.csv', names=header_list)

train_videos = video_files[:int(0.8*len(video_files))]
valid_videos = video_files[int(0.8*len(video_files)):]

print(f"TRAIN: {len(train_videos)}, TEST: {len(valid_videos)}")

train_data = video_dataset(train_videos,labels,sequence_length=10,transform=train_transforms)
val_data = video_dataset(valid_videos,labels,sequence_length=10,transform=test_transforms)

# Lowered batch size to 2 to surely fit in 6GB VRAM along with ResNeXt50 + LSTM
train_loader = DataLoader(train_data, batch_size=2, shuffle=True,  num_workers=2)
valid_loader = DataLoader(val_data,   batch_size=2, shuffle=False, num_workers=2)

# 4. Model Definition
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

# 5. Initialization
print("Initializing model...")
model = Model(2)

print("Loading Checkpoint 1 (77.67% accuracy) to resume and improve...")
try:
    model.load_state_dict(torch.load('checkpoint.pt'))
except Exception as e:
    print(f"Could not load checkpoint.pt: {e}")

if torch.cuda.is_available():
    model = model.cuda()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size

def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (inputs, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            targets = targets.type(torch.cuda.LongTensor).cuda()
            inputs = inputs.cuda()
        _,outputs = model(inputs)
        loss  = criterion(outputs,targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write(
                "\\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Acc: %.2f%%]"
                % (epoch, num_epochs, i, len(data_loader), losses.avg, accuracies.avg))
        sys.stdout.flush()
    print("")
    torch.save(model.state_dict(), './checkpoint_v2.pt')
    return losses.avg,accuracies.avg

def test(epoch, model, data_loader, criterion):
    print('Testing...')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                targets = targets.type(torch.cuda.LongTensor).cuda()
                inputs = inputs.cuda()
            _,outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets))
            acc = calculate_accuracy(outputs, targets)
            _,p = torch.max(outputs,1) 
            true += targets.detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                    "\\r[Batch %d/%d] [Loss: %f, Acc: %.2f%%]"
                    % (i, len(data_loader), losses.avg, accuracies.avg))
            sys.stdout.flush()
        print('\\nAccuracy: %.2f%%' % (accuracies.avg))
    return true,pred,losses.avg,accuracies.avg

lr = 1e-5
num_epochs = 10  # Resuming training for 10 more epochs!
optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay = 1e-5)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

train_loss_avg = []
train_accuracy = []
test_loss_avg = []
test_accuracy = []

print("Starting training...")
for epoch in range(1, num_epochs+1):
    l, acc = train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer)
    train_loss_avg.append(l)
    train_accuracy.append(acc)
    true,pred,tl,t_acc = test(epoch,model,valid_loader,criterion)
    test_loss_avg.append(tl)
    test_accuracy.append(t_acc)

print("Training finished. New weights saved to checkpoint_v2.pt")

plot_epochs = range(1, num_epochs+1)

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(plot_epochs, train_accuracy, 'g', label='Training accuracy')
plt.plot(plot_epochs, test_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.close()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(plot_epochs, train_loss_avg, 'g', label='Training loss')
plt.plot(plot_epochs, test_loss_avg, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.close()

# Save metrics to a file for future plotting without retraining
import json
metrics = {
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
    "train_loss_avg": train_loss_avg,
    "test_loss_avg": test_loss_avg
}
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Saved accuracy_plot.png, loss_plot.png, and training_metrics.json")
