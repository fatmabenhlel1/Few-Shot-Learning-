import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report

# ----------------------------
# Configuration
# ----------------------------
FRAME_COUNT = 48
RESIZE_DIM = (112, 112)
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')
CHECKPOINT_PTH = "model_epoch_10.pth"
NUM_EVAL_CLASSES = 8  # C-way classification
NUM_SHOTS =    1     # k-shot learning

# ----------------------------
# 3D CNN Model Architecture
# ----------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x, return_embeddings=False):
        x = self.features(x)           # [B, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)      # [B, 128]
        if return_embeddings:
            return x
        return self.fc(x)              # [B, out_dim]

# ----------------------------
# Video Dataset Loader
# ----------------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=FRAME_COUNT):
            self.samples = []
            self.num_frames = num_frames
            self.classes = sorted([
                d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            
            if len(self.classes) != NUM_EVAL_CLASSES:
                raise ValueError(f"Expected {NUM_EVAL_CLASSES} classes, found {len(self.classes)}")
            
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            # Collect all video paths (modified to exclude _seg files)
            for cls in self.classes:
                cls_dir = os.path.join(root_dir, cls)
                for fname in os.listdir(cls_dir):
                    # Add exclusion for _seg files
                    if (fname.lower().endswith(VIDEO_EXTS) and 
                        "_seg" not in fname and  # Exclude segmented videos
                        not fname.endswith('.meta')):  # Exclude meta files
                        self.samples.append((
                            os.path.join(cls_dir, fname),
                            self.class_to_idx[cls]
                        ))
            
            if not self.samples:
                raise RuntimeError(f"No valid videos found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_video(path)  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # [C, T, H, W]
        return frames.float(), label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames > 0:
            indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        else:
            indices = np.zeros(self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:  # Handle incomplete videos
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, RESIZE_DIM)
            frames.append(frame)
        cap.release()
        
        # Pad with last frame if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((*RESIZE_DIM, 3)))
        
        # Normalize to [0, 1]
        video = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return video

# ----------------------------
# Few-Shot Evaluation Logic
# ----------------------------
def main():
    # 1. Load pretrained model
    model40 = Simple3DCNN(out_dim=40)
    model40.load_state_dict(torch.load(CHECKPOINT_PTH, map_location=DEVICE))
    
    # 2. Create evaluation model with shared backbone
    model = Simple3DCNN(out_dim=NUM_EVAL_CLASSES)
    model.features = model40.features
    model.to(DEVICE)
    model.eval()

    # 3. Prepare dataset
    dataset = VideoDataset(root_dir="./Data")
    
    # 4. Split into support and query sets
    support_indices, query_indices = [], []
    for cls_idx in range(NUM_EVAL_CLASSES):
        class_samples = [i for i, (_, label) in enumerate(dataset.samples) if label == cls_idx]
        np.random.shuffle(class_samples)
        support_indices.extend(class_samples[:NUM_SHOTS])
        query_indices.extend(class_samples[NUM_SHOTS:NUM_SHOTS+1])
    
    support_set = Subset(dataset, support_indices)
    query_set = Subset(dataset, query_indices)

    # 5. Extract support embeddings
    support_loader = DataLoader(support_set, batch_size=BATCH_SIZE, shuffle=False)
    support_embeddings, support_labels = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(support_loader, desc="Processing Support Set"):
            embeddings = model(videos.to(DEVICE), return_embeddings=True)
            support_embeddings.append(embeddings.cpu())
            support_labels.append(labels)
    
    support_embeddings = torch.cat(support_embeddings).to(DEVICE)
    support_labels = torch.cat(support_labels)

    # 6. Process query set
    query_loader = DataLoader(query_set, batch_size=1, shuffle=False)
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(query_loader, desc="Processing Queries"):
            query_embed = model(videos.to(DEVICE), return_embeddings=True)
            
            # Calculate distances (single query vs all support)
            distances = torch.cdist(query_embed, support_embeddings)
            nearest = torch.argmin(distances)
            pred = support_labels[nearest.cpu()].item()
            
            all_preds.append(pred)
            all_labels.append(labels.item())

    # 7. Generate classification report (updated)
    print("\nFew-Shot Classification Report:")
    print(classification_report(
        all_labels, all_preds, 
        labels=range(NUM_EVAL_CLASSES),  # Explicitly specify all class indices
        target_names=dataset.classes, 
        digits=4,
        zero_division=0
    ))

if __name__ == "__main__":
    main()