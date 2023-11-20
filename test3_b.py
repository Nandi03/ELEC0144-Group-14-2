from transfer_learning import TransferLearning

alexnet_transfer = TransferLearning(
    model_name='alexnet',
    batch_size=5,
    lr=0.0001,
    num_epochs=100
)
alexnet_transfer.train()
