import streamlit as st
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch import nn
from torchvision import models
import tempfile
import os
import subprocess

# Set UI styling and layout configuration
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")

# Hide default Streamlit footer and menu for cleaner UI
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

@st.cache_resource
def load_model(model_path='./checkpoint_v2.pt'):
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
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

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
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        
        while len(frames) < self.count and len(frames) > 0:
            frames.append(frames[-1])
            
        if len(frames) == 0:
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
    
    # --- Class Activation Map (CAM) ---
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    idx = int(prediction.item())
    
    seq_len = fmap.shape[0]
    frame_indices = [0, seq_len // 2, seq_len - 1]
    heatmaps = []
    
    for f_idx in frame_indices:
        curr_fmap = fmap[f_idx].detach().cpu().numpy()
        nc, h, w = curr_fmap.shape
        
        cam = np.dot(curr_fmap.reshape((nc, h*w)).T, weight_softmax[idx,:].T)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / (np.max(cam) + 1e-8)
        cam_img = np.uint8(255 * cam_img)
        
        out_cam = cv2.resize(cam_img, (im_size, im_size))
        heatmap_cv = cv2.applyColorMap(out_cam, cv2.COLORMAP_JET)
        heatmap_cv = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB) # Convert to RGB
        
        original_img = im_convert(img[:, f_idx, :, :, :]) # [H, W, 3]
        
        # Overlay heatmap
        result = heatmap_cv * 0.4/255.0 + original_img * 0.6
        result = result / np.max(result)
        result = np.uint8(255 * result)
        heatmaps.append(result)
    # ----------------------------------
    
    return int(prediction.item()), confidence, heatmaps

# UI Layout
st.markdown("<h1 style='text-align: center; color: white;'>Deepfake Video Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: lightgrey;'>Upload a video clip to see if it is REAL or FAKE</h4>", unsafe_allow_html=True)

st.write("---")

try:
    with st.spinner("Loading Model Checkpoint..."):
        model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an MP4 video clip", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save to a temporary file for model processing
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    # Transcode video to standard H.264 for Streamlit browser compatibility
    display_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    display_tfile.close()
    
    # Use ffmpeg to convert ensuring universal HTML5 compatibility
    subprocess.run(["ffmpeg", "-y", "-i", tfile.name, "-vcodec", "libx264", "-acodec", "aac", display_tfile.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                   
    try:
        with open(display_tfile.name, "rb") as f:
            valid_video_bytes = f.read()
            st.video(valid_video_bytes, format="video/mp4")
    except Exception:
        # Fallback if ffmpeg fails for some reason
        st.video(tfile.name)
        
    try:
        os.remove(display_tfile.name)
    except:
        pass
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        analyze_button = st.button("Analyze Video", use_container_width=True)
        
    if analyze_button:
        with st.spinner("Analyzing video frames and generating activation maps..."):
            dataset = validation_dataset([tfile.name], sequence_length=20, transform=train_transforms)
            video_tensor = dataset[0]
            
            with torch.no_grad():
                pred, conf, heatmap_imgs = predict(model, video_tensor, device)
                
            label = "REAL" if pred == 1 else "FAKE"
            
            st.write("---")
            if label == "REAL":
                st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'> Result: {label}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='text-align: center; color: #F44336;'> Result: {label}</h3>", unsafe_allow_html=True)
                
            st.markdown(f"<h5 style='text-align: center; color: lightgrey;'>Model Confidence: {conf:.2f}%</h5>", unsafe_allow_html=True)
            
            st.write("---")
            st.markdown("<h4 style='text-align: center;'> Explainability Maps over Time</h4>", unsafe_allow_html=True)
            
            if label == "FAKE":
                st.markdown("<p style='text-align: center; color: lightgrey;'>The model identified synthetic manipulation algorithms. The warm/red areas in the heatmaps below track exactly where the model spotted inconsistencies (often around the eyes, mouth, or facial boundaries) across different timestamps of the video.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align: center; color: lightgrey;'>The model found no significant manipulation artifacts. The warm/red areas in the heatmaps indicate the natural facial features the model tracked to confirm the authenticity of the video across different timestamps.</p>", unsafe_allow_html=True)
                
            # Show the heatmaps side-by-side
            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                st.image(heatmap_imgs[0], use_container_width=True, caption="Timestamp: Beginning")
            with hcol2:
                st.image(heatmap_imgs[1], use_container_width=True, caption="Timestamp: Middle")
            with hcol3:
                st.image(heatmap_imgs[2], use_container_width=True, caption="Timestamp: End")
