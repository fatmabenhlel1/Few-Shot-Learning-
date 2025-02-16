import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

# Configuration (from repo config file)
DATA_ROOT = 'd:/Downloads/tieredImageNet/tiered_imagenet'
NUM_CLASSES = 351
BATCH_SIZE = 64
IMAGE_SIZE = 84
BASE_LR = 0.1
LR_GAMMA = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 90
LR_STEPS = [30, 60]
NUM_WAYS = 5
NUM_SHOTS = [1, 6]
NUM_QUERIES = 15
NUM_TASKS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DenseNet-121 Implementation (official architecture)
class DenseNet121(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_dense_block(6, 64, 32),  # 6 layers, 64 input channels, growth rate 32
            self._make_transition(256, 128),    # 256 in -> 128 out
            self._make_dense_block(12, 128, 32),
            self._make_transition(512, 256),
            self._make_dense_block(24, 256, 32),
            self._make_transition(1024, 512),
            self._make_dense_block(16, 512, 32),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(1024, num_classes)

    def _make_dense_block(self, num_layers, in_channels, growth_rate):
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv(nn.functional.relu(self.bn(x)))
        return torch.cat([x, out], 1)

# Data Transforms (official preprocessing)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloader(split):
    return DataLoader(
        ImageFolder(os.path.join(DATA_ROOT, split), 
                   transform=train_transform if split == "train" else test_transform),
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True
    )

def train_model():
    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")
    
    model = DenseNet121().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=LR_GAMMA)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "densenet121_tiered.pth")
        
        scheduler.step()
    
    return model

from sklearn.metrics import classification_report

def evaluate_fewshot(model):
    test_loader = get_dataloader("test")
    train_loader = get_dataloader("train")
    
    # Get base class mean
    model.eval()
    train_features = []
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader, desc="Extracting base features"):
            train_features.append(model(inputs.to(DEVICE)).cpu())
    train_features = torch.cat(train_features)
    mean_feature = train_features.mean(dim=0)

    def apply_transforms(features, transform):
        if transform == "L2N":
            return features / features.norm(p=2, dim=1, keepdim=True)
        if transform == "CL2N":
            centered = features - mean_feature
            return centered / centered.norm(p=2, dim=1, keepdim=True)
        return features

    results = {}
    for num_shots in NUM_SHOTS:
        print(f"\nEvaluating {num_shots}-shot:")
        
        # Initialize lists to store predictions and labels
        all_preds = {t: [] for t in ["UN", "L2N", "CL2N"]}
        all_labels = []

        for _ in tqdm(range(NUM_TASKS), desc="Tasks"):
            # Sample task
            class_indices = np.random.choice(len(test_loader.dataset.classes), NUM_WAYS, replace=False)
            support, query = [], []
            
            for c in class_indices:
                samples = [i for i, (_, y) in enumerate(test_loader.dataset.samples) if y == c]
                selected = np.random.choice(samples, num_shots + NUM_QUERIES, replace=False)
                support.extend(selected[:num_shots])
                query.extend(selected[num_shots:])
            
            # Extract features
            with torch.no_grad():
                sup_inputs = torch.stack([test_transform(Image.open(test_loader.dataset.samples[i][0]).convert("RGB")) for i in support])
                qry_inputs = torch.stack([test_transform(Image.open(test_loader.dataset.samples[i][0]).convert("RGB")) for i in query])
                
                sup_features = model(sup_inputs.to(DEVICE)).cpu()
                qry_features = model(qry_inputs.to(DEVICE)).cpu()
            
            # Calculate centroids
            centroids = torch.stack([sup_features[i*num_shots:(i+1)*num_shots].mean(0) 
                                   for i in range(NUM_WAYS)])
            
            # Evaluate transforms
            for transform in ["UN", "L2N", "CL2N"]:
                t_centroids = apply_transforms(centroids, transform)
                t_qry = apply_transforms(qry_features, transform)
                
                dists = torch.cdist(t_qry, t_centroids)
                preds = dists.argmin(dim=1).numpy()
                all_preds[transform].extend(preds)
            
            # Store labels
            labels = torch.arange(NUM_WAYS).repeat_interleave(NUM_QUERIES).numpy()
            all_labels.extend(labels)
        
        # Generate classification report for each transform
        print(f"\n{num_shots}-shot Classification Report:")
        for transform in ["UN", "L2N", "CL2N"]:
            print(f"\nTransform: {transform}")
            print(classification_report(
                all_labels,
                all_preds[transform],
                target_names=[f"Class {i}" for i in range(NUM_WAYS)],
                digits=4
            ))

if __name__ == "__main__":
    # Train or load model
    if not os.path.exists("densenet121_tiered.pth"):
        print("Training DenseNet-121...")
        model = train_model()
    else:
        print("Loading pretrained model...")
        model = DenseNet121(num_classes=None).to(DEVICE)
        state_dict = torch.load("densenet121_tiered.pth", map_location=DEVICE)
        # Filter classifier weights
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        model.load_state_dict(state_dict, strict=False)
    
    # Remove classifier for feature extraction
    model.classifier = nn.Identity()
    
    # Evaluate
    evaluate_fewshot(model)