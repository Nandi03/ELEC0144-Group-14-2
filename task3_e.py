from transfer_learning import TransferLearning

googlenet_transfer = TransferLearning(
    model_name='googlenet',
    optimiser="adam",
    batch_size=30,
    lr=0.0001,
    num_epochs=10,
    datasetMode="single"
)
googlenet_transfer.train()