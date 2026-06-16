import torch
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader

# setting device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# loading model
model = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vits14'
)

model = model.to(device)
model.eval()

# preprocessing
tranform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# loading dataset
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tranform)

# creating dataloader
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
)

# generating embeddings
all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        embeddings = model(images)
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.append(labels.numpy())

# merging all embeddings and labels
all_embeddings = np.concatenate(all_embeddings, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# checking the shapes
print(f"Embeddings shape: {all_embeddings.shape}")
print(f"Labels shape: {all_labels.shape}")

# saving
np.save(
    'embeddings/cifar10_embeddings.npy',
    all_embeddings
)
np.save(
    'embeddings/cifar10_labels.npy',
    all_labels
)