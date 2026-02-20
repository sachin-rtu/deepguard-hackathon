# ==========================================
# test_models.py - MODEL COMPARISON SUITE
# ==========================================
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔬 Testing on device: {device}")

# ==========================================
# 1. ARCHITECTURES
# ==========================================
# Baseline ResNet18 (Spatial only)
class BaselineResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )
    def forward(self, x):
        return self.model(x)

# Advanced Temporal Model (CNN + BiLSTM + Transformer)
class AdvancedDeepfakeDetector(nn.Module):
    def __init__(self, num_frames=8, hidden_size=256, num_layers=2):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        lstm_out_dim = hidden_size * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_out_dim, nhead=8, dim_feedforward=512, dropout=0.4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2)
        )
    
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        features = [self.feature_extractor(x[:, i]).view(batch_size, -1) for i in range(num_frames)]
        features = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(features)
        transformer_out = self.transformer(lstm_out)
        return self.classifier(torch.mean(transformer_out, dim=1))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(frame)))
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(torch.zeros((3, 224, 224)))
    return torch.stack(frames[:num_frames]).unsqueeze(0).to(device)

def predict(model, tensor_input, is_temporal=False):
    with torch.no_grad():
        if not is_temporal:
            # Baseline just looks at the middle frame
            tensor_input = tensor_input[:, 4, :, :, :] 
        outputs = model(tensor_input)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        verdict = 'Fake' if pred.item() == 1 else 'Real'
        return {'verdict': verdict, 'confidence': conf.item()}

# ==========================================
# 3. EXECUTION LOGIC (FIXED LOADING)
# ==========================================
print("📥 Loading models...")
baseline_model = BaselineResNet().to(device)
temporal_model = AdvancedDeepfakeDetector().to(device)

# 1. Try to load Baseline
try:
    # IMPORTANT: Update this path if you uploaded it to Kaggle!
    baseline_model.load_state_dict(torch.load('/kaggle/input/YOUR_DATASET_NAME/deepfake_resnet18_t4.pth', map_location=device))
    print("✅ Baseline model loaded.")
except Exception as e:
    print("⚠️ Baseline model missing. It will guess randomly (expect ~50%).")

# 2. Try to load Temporal
try:
    temporal_model.load_state_dict(torch.load('/kaggle/working/temporal_deepfake_detector_best.pth', map_location=device))
    print("✅ Temporal model loaded successfully!")
except Exception as e:
    print(f"⚠️ Temporal load failed: {e}")

baseline_model.eval()
temporal_model.eval()

# Grab 25 Real and 25 Fake videos
test_videos = list(Path('/kaggle/working/dataset/original_sequences').rglob('*.mp4'))[:25] + \
              list(Path('/kaggle/working/dataset/manipulated_sequences').rglob('*.mp4'))[:25]

print(f"🎬 Testing on {len(test_videos)} videos...")

results = []
for video in test_videos:
    actual = 'Real' if 'original' in str(video) else 'Fake'
    fake_type = video.parent.parent.name if actual == 'Fake' else 'N/A'
    
    frames_tensor = extract_frames(video)
    
    base_res = predict(baseline_model, frames_tensor, is_temporal=False)
    temp_res = predict(temporal_model, frames_tensor, is_temporal=True)
    
    results.append({
        'video_name': video.name,
        'actual': actual,
        'fake_type': fake_type,
        'baseline_pred': base_res['verdict'],
        'baseline_conf': base_res['confidence'],
        'temporal_pred': temp_res['verdict'],
        'temporal_conf': temp_res['confidence'],
        'baseline_correct': actual == base_res['verdict'],
        'temporal_correct': actual == temp_res['verdict']
    })


# ==========================================
# 4. METRICS & EXPORT
# ==========================================
df = pd.DataFrame(results)
df.to_csv('/kaggle/working/model_comparison.csv', index=False)

baseline_acc = df['baseline_correct'].mean() * 100
temporal_acc = df['temporal_correct'].mean() * 100

print("\n" + "="*40)
print("📊 TEST SUITE RESULTS")
print("="*40)
print(f"Baseline (ResNet) Accuracy: {baseline_acc:.1f}%")
print(f"Temporal (Hybrid) Accuracy: {temporal_acc:.1f}%")
print(f"🔥 Overall Improvement:     +{temporal_acc - baseline_acc:.1f}%")
print("="*40)
print("✅ Results exported to model_comparison.csv")