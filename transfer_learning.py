import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet_Weights, GoogLeNet_Weights
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

class TransferLearning:
    def __init__(self, model_name, optimiser,batch_size, datasetMode,lr=0.01, num_classes=5, num_epochs=100, 
                criterion=nn.CrossEntropyLoss(), momentum=0.9, num_layers_to_replace=3, train_test_split="70:30"):
        
        '''
        Initializes the TransferLearning class with the specified parameters.

        Args:
            model_name (str): Name of the pre-trained model ('alexnet' or 'googlenet').
            optimiser (str): Type of optimiser to use ('adam' or 'sgdm').
            batch_size (int): Batch size for training and validation.
            datasetMode (str): Mode of the dataset ('single' or 'double').
            lr (float): Learning rate for the optimiser.
            num_classes (int): Number of classes in the classification task.
            num_epochs (int): Number of training epochs.
            criterion: Loss function.
            momentum (float): Momentum for the SGD optimiser.
            num_layers_to_replace (int): Number of classifier layers to replace in the modified AlexNet.

        Returns:
            None
        '''
       
        self.class_names = ["Durian", "Kiwi", "Mango", "Mangosteen", "Papaya"]
        self.model_name = model_name
        self.optimiser = optimiser
        self.criterion = criterion
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.num_layers_to_replace = num_layers_to_replace
        self.num_epochs = num_epochs
        self.datasetMode = datasetMode
        self.train_test_split = train_test_split
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
        self._use_pretrained_model()

        # Set up optimiser and loss function
        self._set_optimiser()

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
            loss_values, accuracy_values = self._train_and_evaluate()

            # Display line graph for loss over epochs
            self._plot_loss_graph(loss_values)

            # Display line graph for accuracy over epochs
            self._plot_accuracy_graph(accuracy_values)

            # Display the table after training using the new method
            self._display_epoch_table(list(zip(range(1, self.num_epochs + 1), loss_values, accuracy_values)))

            self._print_confusion_matrix()
            # Save the trained model
            # torch.save(self.model.state_dict(), f'fruit_classifier_{self.model_name}.pth')


    def _train_and_evaluate(self):
        '''
        Internal method for training the model and evaluating loss and accuracy.

        Args:
            None

        Returns:
            list: Loss values for each epoch.
            list: Accuracy values for each epoch.
        '''
        loss_values = []  # List to store loss for each epoch
        accuracy_values = []  # List to store accuracy for each epoch

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimiser.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()

            # Validation
            accuracy = self._evaluate()

            # Append loss and accuracy values for the current epoch
            loss_values.append(loss.item())
            accuracy_values.append(accuracy)

        return loss_values, accuracy_values

    def _set_optimiser(self):
        '''
        Sets the optimiser for the model.

        Args:
            None

        Returns:
            None
        '''
        if self.optimiser == "adam":
            self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimiser == "sgdm":
            self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def _use_pretrained_model(self):
        '''
        Sets the model to either AlexNet or GoogLeNet and modifies the classifier layers for AlexNet.

        Args:
            model_name (str): Name of the pre-trained model ('alexnet' or 'googlenet').
            num_layers_to_replace (int): Number of classifier layers to replace in the modified AlexNet.

        Returns:
            None
        '''
        if self.model_name == 'alexnet':
            self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
            self._modify_alexnet(self.num_layers_to_replace)
        elif self.model_name == 'googlenet':
            self.model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

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
        Loads and prepares the training and validation data based on the specified dataset mode.

        Args:
            None

        Returns:
            DataLoader: Training data loader.
            DataLoader: Validation data loader.
        '''

        if self.datasetMode == "single":
            if self.train_test_split == "70:30":
                train_path = "task3_70:30_split/single/train"
                test_path = "task3_70:30_split/single/test"

            elif self.train_test_split == "60:40":
                train_path = "task3_60:40_split/single/train"
                test_path = "task3_60:40_split/single/test"

        elif self.datasetMode == "multiple":
            if self.train_test_split == "70:30":
                train_path = "task3_70:30_split/multiple/train"
                test_path = "task3_70:30_split/multiple/test"

            elif self.train_test_split == "60:40":
                train_path = "task3_60:40_split/multiple/train"
                test_path = "task3_60:40_split/multiple/test"

        # Load training data
        train_dataset = datasets.ImageFolder(train_path, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        # Load validation data
        val_dataset = datasets.ImageFolder(test_path, transform=self.transform)
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

    def _print_confusion_matrix(self):
        '''
        Prints the confusion matrix after training.

        Args:
            None

        Returns:
            None
        '''

        self.model.eval()
        all_labels = []
        all_predicted = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predicted)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def _display_epoch_table(self, table_data):
        '''
        Displays a table with epoch, loss, and accuracy using matplotlib.

        Args:
            table_data (list): List containing lists of epoch, loss, and accuracy for each epoch.

        Returns:
            None
        '''
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        headers = ['Epoch', 'Loss', 'Accuracy']
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center', colColours=['#f3f3f3']*3)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.show()

    def _plot_loss_graph(self, loss_values):
        '''
        Plots a line graph for loss over epochs.

        Args:
            loss_values (list): Loss values for each epoch.

        Returns:
            None
        '''
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.num_epochs + 1), loss_values, label='Loss', marker='o')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def _plot_accuracy_graph(self, accuracy_values):
        '''
        Plots a line graph for accuracy over epochs.

        Args:
            accuracy_values (list): Accuracy values for each epoch.

        Returns:
            None
        '''
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.num_epochs + 1), accuracy_values, label='Accuracy', marker='o', color='green')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.show()
