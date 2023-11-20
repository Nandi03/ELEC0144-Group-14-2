import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet, GoogLeNet

# Set random seed for reproducibility
torch.manual_seed(42)

class TransferLearning:
    def __init__(self, model_name, batch_size, lr, num_classes=5, train_path = "task3data/train", test_path = "task3data/test", num_epochs=100):
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.train_path = train_path
        self.test_path = test_path
        self.input_size = (227,227)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Move to gpu if available

        # Data augmentation and normalization
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the data
        self.train_loader, self.val_loader = self.load_data(self.train_path, self.test_path, transform, self.batch_size)

        # Load pre-trained model
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        elif model_name == 'googlenet':
            self.model = models.googlenet(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Move model to device
        self.model = self.model.to(self.device)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Validation
            accuracy = self.evaluate()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{self.num_epochs}, Accuracy: {accuracy}%')

        # Save the trained model
        torch.save(self.model.state_dict(), f'fruit_classifier_{self.model_name}.pth')

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
    
    def load_data(self,train_path, test_path, transform, batch_size):
        # Load training data
        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Load validation data
        val_dataset = datasets.ImageFolder(test_path, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
