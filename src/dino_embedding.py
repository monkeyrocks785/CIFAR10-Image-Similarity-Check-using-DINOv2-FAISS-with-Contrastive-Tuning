import torch
from torchvision import transforms, datasets

# setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# loading dinov2 model

model = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vits14'
)

model = model.to(device)
model.eval()

# dinov2 preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# loading dataset

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

image, label = dataset[0]
print(f'Image shape: {image.shape}, Label: {label}')

# add batch dimension

image = image.unsqueeze(0).to(device)

# get embedding

with torch.no_grad():
    embedding = model(image)

print(f'Embedding shape: {embedding.shape}')
print(f'First 10 elements of embedding: {embedding[0][:10]}')