import matplotlib.pyplot as plt
import japanize_matplotlib
import settings


def get_loss_data(file):
    loss = {'train': [], 'valid': []}
    with open(file, 'r')as f:
        data = f.read().splitlines()
    for d in data:
        info = d.split()
        if len(info) == 9:
            loss['train'].append(float(info[2]))
            loss['valid'].append(float(info[6]))
    return loss


def plt_loss(loss):
    epochs = list(range(1, len(loss['train'])+1))
    for key, value in loss.items():
        plt.plot(epochs, value)
    plt.legend(loss.keys())
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.show()
    return


def main():
    loss = get_loss_data(settings.LOG_FILE)
    plt_loss(loss)
    return


if __name__ == '__main__':
    main()
