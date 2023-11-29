from transfer_learning import TransferLearning

googlenet_transfer = TransferLearning(
    model_name='googlenet',
    optimiser="adam",
    batch_size=30,
    lr=0.0005,
    num_epochs=10,
    datasetMode="single",
    train_test_split="60:40" 
)
googlenet_transfer.train()
