import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import GoogLeNet_Weights

# Set random seed for reproducibility
torch.manual_seed(42)

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Inception v1 expects 224x224 input
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the data
train_dataset = datasets.ImageFolder('task3data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = datasets.ImageFolder('task3data/test', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Load pre-trained GoogleNet (Inception v1)
googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)

# Modify the classifier
num_fruit_classes = 5
googlenet.fc = nn.Linear(googlenet.fc.in_features, num_fruit_classes)

# Set up optimizer and loss function
momentum = 0.9
optimizer = optim.Adam(googlenet.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
googlenet = googlenet.to(device)

for epoch in range(1, num_epochs + 1):
    googlenet.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = googlenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    googlenet.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = googlenet(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    if epoch % 1 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%')

# Save the trained model
torch.save(googlenet.state_dict(), 'fruit_classifier_googlenet.pth')
