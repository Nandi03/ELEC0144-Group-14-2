from transfer_learning import TransferLearning

# modify 3b parameters
alexnet_transfer_3b = TransferLearning(
    model_name='alexnet',
    optimiser="adam",
    batch_size=5,
    lr=0.0001,
    num_epochs=10,
    datasetMode="single"
)
alexnet_transfer_3b.train()

# modify 3c parameters
alexnet_transfer_3c = TransferLearning(
    model_name='alexnet',
    optimiser="adam",
    batch_size=5,
    lr=0.0001,
    num_epochs=10,
    num_classes=4,
    datasetMode="single"
)
alexnet_transfer_3c.train()

# modify 3d paramters
googlenet_transfer_3d = TransferLearning(
    model_name='googlenet',
    optimiser="adam",
    batch_size=30,
    lr=0.0001,
    num_epochs=10,
    datasetMode="single"
)
googlenet_transfer_3d.train()

# modify 3e paramters
googlenet_transfer_3e = TransferLearning(
    model_name='googlenet',
    optimiser="adam",
    batch_size=30,
    lr=0.0001,
    num_epochs=10,
    datasetMode="single"
)
googlenet_transfer_3e.train()