from transfer_learning import TransferLearning

traindata = "task3data/train"
testdata =  "task3data/test"

alexnet_transfer = TransferLearning(
    model_name='alexnet',
    optimiser="adam",
    batch_size=5,
    lr=0.0001,
    num_epochs=10,
    datasetMode="single",
    num_layers_to_replace=4
)
alexnet_transfer.train()
