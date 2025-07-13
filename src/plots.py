import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss', color='aqua')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss', color='yellow')

    plt.title('Epochs vs loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_predictions(y_test, y_pred):
    sns.scatterplot(x=y_test, y=y_pred, color='aqua', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Actual')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.legend()
    plt.show()
    

