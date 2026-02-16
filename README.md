# DeepGuard - AI Deepfake Detection

Hackathon project using temporal deep learning for deepfake detection.

## Status: In Development
- ✅ ResNet18 baseline model trained (85-90% accuracy)
- ✅ Gradio demo working
- 🚧 Training temporal LSTM model (target: 94% accuracy)
- 🚧 UI improvements in progress
- 🚧 Documentation in progress

## Team
- [Your Name] - ML/Model Training
- [Teammate 1] - UI/Frontend
- [Teammate 2] - Research/Documentation

## How to Run
````bash
pip install -r requirements.txt
python Gradio.py
````

## Tasks
See [TASKS.md](TASKS.md) for current sprint

## Branches
- `main` - stable code only
- `ui-improvements` - UI teammate work here
- `model-training` - Model experiments
- `documentation` - Research/docs
````

**requirements.txt**
````
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
plotly>=5.18.0
