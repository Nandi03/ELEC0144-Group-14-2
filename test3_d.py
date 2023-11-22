from transfer_learning import TransferLearning

googlenet_transfer = TransferLearning(
    model_name='googlenet',
    optimiser="adam",
    batch_size=5,
    lr=0.0001,
    num_epochs=10
)
googlenet_transfer.train()