import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet_Weights, GoogLeNet_Weights

# Set random seed for reproducibility
torch.manual_seed(42)

class TransferLearning:
    def __init__(self, model_name, optimiser,batch_size, lr=0.01, num_classes=5, train_path = "task3data/train", test_path = "task3data/test", num_epochs=100, criterion=nn.CrossEntropyLoss(), momentum=0.9, num_layers_to_replace=1):
        
        '''
        Initializes the TransferLearning class with the specified parameters.

        Args:
            model_name (str): Name of the pre-trained model ('alexnet' or 'googlenet').
            optimizer (str): Name of the optimizer ('adam' or 'sgdm').
            batch_size (int): Number of samples in each batch for training and validation.
            lr (float): Learning rate for the optimizer.
            num_classes (int): Number of classes in the classification task.
            train_path (str): Path to the training data.
            test_path (str): Path to the validation data.
            num_epochs (int): Number of epochs for training.
            criterion (torch.nn.Module): Loss function for training.
            momentum (float): Momentum factor for the SGD optimizer.
            num_layers_to_replace (int): Number of classifier layers to replace in the modified AlexNet.

        Attributes:
            model_name (str): Name of the pre-trained model.
            optimizer (str): Name of the optimizer.
            batch_size (int): Number of samples in each batch.
            lr (float): Learning rate for the optimizer.
            num_classes (int): Number of classes.
            num_layers_to_replace (int): Number of layers to replace in the modified AlexNet.
            num_epochs (int): Number of training epochs.
            train_path (str): Path to the training data.
            test_path (str): Path to the validation data.
            criterion (torch.nn.Module): Loss function.
            momentum (float): Momentum factor for SGD optimizer.
            transform (torchvision.transforms.Compose): Data augmentation and normalization transformations.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device to which the model is moved (GPU or CPU).
            model (torch.nn.Module): Pre-trained model with modified classifier.
            optimizer (torch.optim.Optimizer): Optimizer for training.

        Returns:
            None
        '''

        self.model_name = model_name
        self.criterion = criterion
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers_to_replace = num_layers_to_replace
        self.num_epochs = num_epochs
        self.train_path = train_path
        self.test_path = test_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # Move to gpu if available

        # Data augmentation and normalization
        self.transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the data
        self.train_loader, self.val_loader = self._load_data()

        # Load pre-trained model
        if model_name == 'alexnet':
            self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
            self._modify_alexnet(self.num_layers_to_replace)

        elif model_name == 'googlenet':
            self.model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # Set up optimizer and loss function
        if self.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "sgdm":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum)


        # Move model to device
        self.model = self.model.to(self.device)

    def train(self):

        '''
        Trains the model for the specified number of epochs.

        Args:
            None

        Returns:
            None
        '''

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
            accuracy = self._evaluate()

            if epoch % 1 == 0:
                print(f'Epoch {epoch}/{self.num_epochs}, Loss: {loss},Accuracy: {accuracy}%')

        # Save the trained model
        # torch.save(self.model.state_dict(), f'fruit_classifier_{self.model_name}.pth')

    def _evaluate(self):

        '''
        Evaluates the model on the validation data and returns the accuracy.

        Args:
            None

        Returns:
            float: Accuracy on the validation data.
        '''

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
    
    def _load_data(self):

        '''
        Loads and prepares the training and validation data using ImageFolder.

        Args:
            None

        Returns:
            torch.utils.data.DataLoader: DataLoader for training data.
            torch.utils.data.DataLoader: DataLoader for validation data.
        '''

        # Load training data
        train_dataset = datasets.ImageFolder(self.train_path, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        # Load validation data
        val_dataset = datasets.ImageFolder(self.test_path, transform=self.transform)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
    
    def _modify_alexnet(self, num_layers_to_replace):

        '''
        Modifies the AlexNet model by replacing a specified number of classifier layers.

        Args:
            num_layers_to_replace (int): Number of layers to replace.

        Returns:
            None
        '''
        # Replace the classifier layers before the last two layers
        
        # Extract the classifier layers
        classifier_layers = list(self.model.classifier.children())

        # Replace the specified number of layers before the last two layers
        modified_classifier = nn.Sequential(*classifier_layers[:-num_layers_to_replace], nn.Linear(4096, self.num_classes))

        # Set the modified classifier back to the model
        self.model.classifier = modified_classifier
