from transfer_learning import TransferLearning

# Example usage for GoogLeNet
googlenet_transfer = TransferLearning(
    model_name='googlenet',
    batch_size=2,
    lr=0.0001,
    num_epochs=100
)
googlenet_transfer.train()