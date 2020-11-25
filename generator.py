from keras.datasets import mnist
import matplotlib.pyplot as plt


def main():
    print("Hello world!")


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f'train data: {x_train.shape}, {y_train.shape}')
    print(f'test data: {x_test.shape}, {y_test.shape}')

    plt.imshow(x_train[5], cmap='gray_r')
    plt.show()
