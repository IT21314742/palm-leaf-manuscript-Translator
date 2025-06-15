import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# --- Dataset Loader ---
class ManuscriptOCRDataset(Dataset):
    def __init__(self, img_dir, label_file=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # For now, label_file is ignored (need to add ground truth labels for real training)
        self.labels = None
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        # Placeholder: return dummy label
        label = "<label>"
        return image, label

# --- Simple CRNN Model (placeholder) ---
class SimpleCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((32, 32))  # output: (batch, 64, 32, 32)
        )
        self.rnn = nn.LSTM(64*32, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128*2, num_classes)
    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        # Debug: print(f"CNN output shape: {x.shape}")
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c*h)
        # Now input_size for LSTM is c*h = 64*32 = 2048
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# --- Training Loop (placeholder) ---
def train_ocr_model(train_dir, num_classes=100, epochs=10, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor()
    ])
    dataset = ManuscriptOCRDataset(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleCRNN(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss()
    for epoch in range(epochs):
        for images, labels in dataloader:
            # Placeholder: No real labels yet
            logits = model(images)
            # ... (real training code with ground truth labels needed)
    # Save model
    torch.save(model.state_dict(), 'ocr_model.pth')
    print('Model saved as ocr_model.pth')

# --- Inference (placeholder) ---
def ocr_infer(image_np):
    # Load model
    model = SimpleCRNN(num_classes=100)
    if os.path.exists('ocr_model.pth'):
        model.load_state_dict(torch.load('ocr_model.pth', map_location='cpu'))
    model.eval()
    # Preprocess
    image = Image.fromarray(image_np)
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
    # Placeholder: return dummy text
    return "<ocr output>"

if __name__ == "__main__":
    # Example: train model on training_dataset/sinhala
    train_ocr_model('dataset/training_dataset/sinhala')
