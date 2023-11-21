from transfer_learning import TransferLearning

alexnet_transfer = TransferLearning(
    model_name='alexnet',
    optimiser="adam",
    batch_size=5,
    lr=0.0001,
    num_epochs=10,
    num_layers_to_replace=1
)
alexnet_transfer.train()
