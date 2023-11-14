import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Assuming your dataset is structured as: root/durian/durian_1.jpg
data_dir = 'task3data'

# Load the dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

# Calculate the size of training and validation sets
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = total_size - train_size

# Split the dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders for training and validation sets
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8),
    'val': DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8),
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

# Load pre-trained GoogLeNet
model = models.googlenet(weights="GoogLeNet_Weights.DEFAULT")
# Change the output layer to match the number of classes
num_fruits = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_fruits)

# Specify loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
num_epochs = 10
device = torch.device("cpu")
model = model.to(device)

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")
