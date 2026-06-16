from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print(f"Total Images : {len(dataset)}")

image, label = dataset[0]

print(f"Image Shape : {image.shape}")
print(f"Label : {label}")

print(f"Class Names : {dataset.classes}")