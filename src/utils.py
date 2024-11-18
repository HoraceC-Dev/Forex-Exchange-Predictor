import matplotlib.pyplot as plt

def graph_generator(val_loss_data,train_loss_data,learning_rates):
    plt.plot(train_loss_data, label="Training Loss")
    plt.plot(val_loss_data, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("graphs/Train&Val_Loss_vs_Epochs.png") 
    plt.clf()
    # plt.plot(learning_rates, label="Learning Rate")
    # plt.xlabel("Epochs")
    # plt.ylabel("Learning Rate")
    # plt.legend()
    # plt.title("Learning Rate Over Epochs")
    # plt.savefig("graphs/LearningRate_vs_Epochs.png") 