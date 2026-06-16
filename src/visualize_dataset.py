import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# loading CIFAR10

dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# using one image

image, label = dataset[0]

print(f"Image shape: {image.shape}")
print(f"Label: {label}")
print(f"Class name: {dataset.classes[label]}")

# converting the image from (C, H, W) to (H, W, C)
image = image.permute(1, 2, 0)

# visualizing the image

plt.imshow(image)
plt.title(f"Class: {dataset.classes[label]}")
plt.axis('off')
plt.show()