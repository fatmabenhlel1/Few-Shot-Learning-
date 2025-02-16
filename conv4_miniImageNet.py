import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Configuration
DATA_ROOT = "./miniImageNet"
IMAGE_SIZE = 84
BATCH_SIZE = 512
TRAIN_EPOCHS = 100
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WAYS = 5
NUM_SHOTS = [1, 5]  # Evaluate both 1-shot and 5-shot
NUM_QUERIES = 15
NUM_TASKS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# Data Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize([int(IMAGE_SIZE * 1.15), int(IMAGE_SIZE * 1.15)]),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Conv4(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 5 * 5, num_classes) if num_classes is not None else None

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

# Data Loading
def get_dataloader(split, transform):
    dataset = ImageFolder(root=os.path.join(DATA_ROOT, split), transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == "train"), num_workers=4)

# Training Function
def train_model():
    train_loader = get_dataloader("train", train_transform)
    model = Conv4(len(train_loader.dataset.classes)).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 66], gamma=0.1)


    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(TRAIN_EPOCHS):
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "conv4_miniimagenet.pth")
    return model

# Feature Extraction
def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            features.append(model(inputs.to(DEVICE)).cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

def evaluate_fewshot(model):
    test_loader = get_dataloader("test", test_transform)
    train_loader = get_dataloader("train", test_transform)
    
    # Get base class mean
    train_features, _ = extract_features(model, train_loader)
    mean_feature = train_features.mean(dim=0)
    
    # Feature transforms
    def apply_transforms(features, transform_type):
        if transform_type == "L2N":
            return features / torch.norm(features, p=2, dim=1, keepdim=True)
        elif transform_type == "CL2N":
            centered = features - mean_feature
            return centered / torch.norm(centered, p=2, dim=1, keepdim=True)
        return features
    
    # Evaluate for both 1-shot and 5-shot
    results = {}
    for num_shots in NUM_SHOTS:
        accuracies = {"UN": [], "L2N": [], "CL2N": []}
        print(f"\nEvaluating {num_shots}-shot {NUM_WAYS}-way classification:")
        
        for _ in tqdm(range(NUM_TASKS)):
            # Sample task
            class_indices = np.random.choice(len(test_loader.dataset.classes), NUM_WAYS, replace=False)
            support_set, query_set = [], []
            
            for c in class_indices:
                class_mask = [i for i, (_, y) in enumerate(test_loader.dataset.samples) if y == c]
                selected = np.random.choice(class_mask, num_shots + NUM_QUERIES, replace=False)
                support_set.extend(selected[:num_shots])
                query_set.extend(selected[num_shots:])
            
            # Extract features (convert grayscale to RGB)
            support_inputs = torch.stack([
                test_transform(Image.open(test_loader.dataset.samples[i][0]).convert("RGB"))  # Convert to RGB
                for i in support_set
            ])
            query_inputs = torch.stack([
                test_transform(Image.open(test_loader.dataset.samples[i][0]).convert("RGB"))  # Convert to RGB
                for i in query_set
            ])
            
            with torch.no_grad():
                support_features = model(support_inputs.to(DEVICE)).cpu()
                query_features = model(query_inputs.to(DEVICE)).cpu()
            
            # Calculate centroids (mean of support features)
            support_labels = torch.arange(NUM_WAYS).repeat_interleave(num_shots)
            centroids = torch.stack([support_features[support_labels == c].mean(0) for c in range(NUM_WAYS)])
            
            # Evaluate transforms
            for transform in ["UN", "L2N", "CL2N"]:
                t_centroids = apply_transforms(centroids, transform)
                t_queries = apply_transforms(query_features, transform)
                
                distances = torch.cdist(t_queries, t_centroids)
                predictions = distances.argmin(dim=1)
                accuracy = (predictions == torch.arange(NUM_WAYS).repeat_interleave(NUM_QUERIES)).float().mean()
                accuracies[transform].append(accuracy.item())
        
        # Store results
        results[num_shots] = {
            transform: (np.mean(accs) * 100, 1.96 * np.std(accs)/np.sqrt(NUM_TASKS) * 100)
            for transform, accs in accuracies.items()
        }
    
    # Print results in paper format
    for num_shots in NUM_SHOTS:
        print(f"\n{num_shots}-shot Results:")
        for transform in ["UN", "L2N", "CL2N"]:
            mean, conf = results[num_shots][transform]
            print(f"{transform}: {mean:.2f} ± {conf:.2f}%")

# Main Execution
if __name__ == "__main__":
    # Train/load model
    if not os.path.exists("conv4_miniimagenet.pth"):
        print("Training Conv-4 model...")
        train_loader = get_dataloader("train", train_transform)
        model = Conv4(len(train_loader.dataset.classes)).to(DEVICE)
        model = train_model()
    else:
        print("Loading pretrained model...")
        model = Conv4(None).to(DEVICE)  # No classifier needed for feature extraction
        
        # Load state dict and filter out classifier weights
        state_dict = torch.load("conv4_miniimagenet.pth", map_location=DEVICE)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        model.load_state_dict(state_dict, strict=False)
    
    # Evaluate
    evaluate_fewshot(model)