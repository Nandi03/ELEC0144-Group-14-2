from transfer_learning import TransferLearning

googlenet_transfer = TransferLearning(
    model_name='googlenet',
    optimiser="adam",
    batch_size=30,
    lr=0.0001,
    num_epochs=10,
    datasetMode="multiple",
    train_test_split="70:30" 
)
googlenet_transfer.train()