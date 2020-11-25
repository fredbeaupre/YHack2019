from keras.datasets import mnist
import matplotlib.pyplot as plt


def main():
    print("Hello world!")


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f'train data: {x_train.shape}, {y_train.shape}')
    print(f'test data: {x_test.shape}, {y_test.shape}')

    for i in range(100):
        plt.subplot(10, 10, 1 + i)
        plt.axis('off')
        plt.imshow(x_train[i], cmap='gray_r')
    plt.show()
