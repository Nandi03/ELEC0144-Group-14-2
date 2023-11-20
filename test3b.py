from transfer_learning import TransferLearning

# Example usage for AlexNet
alexnet_transfer = TransferLearning(
    model_name='alexnet',
    num_classes=5,
    input_size=(227, 227),
    train_path='task3data/train',
    test_path='task3data/test',
    batch_size=5,
    lr=0.0001,
    num_epochs=100
)
alexnet_transfer.train()

# Example usage for GoogLeNet
googlenet_transfer = TransferLearning(
    model_name='googlenet',
    num_classes=5,
    input_size=(224, 224),
    train_path='task3data/train',
    test_path='task3data/test',
    batch_size=2,
    lr=0.0001,
    num_epochs=100
)
googlenet_transfer.train()