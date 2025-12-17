import matplotlib.pyplot as plt


def plotLoss(_trainingLoss, _evalLoss):
    epochs =  range(1, len(_trainingLoss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, _trainingLoss, label="Training Loss", marker='o')
    plt.plot(epochs, _evalLoss, label="Validation Loss", marker='s')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.show()