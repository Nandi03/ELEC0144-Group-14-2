from transfer_learning import TransferLearning

alexnet_transfer = TransferLearning(
    model_name='alexnet',
    optimiser="adam",
    batch_size=5,
    lr=0.0001,
    num_epochs=10,
    datasetMode="multiple",
    train_test_split="70:30" 
)
alexnet_transfer.train()
