from struct import unpack
from array import array
import numpy as np
import matplotlib.pyplot as plt
from Net import Net


def load_image(url, validation_code):
    with open(url, 'rb') as file:
        magic_number, number_of_images, image_height, image_width = unpack(">IIII", file.read(16))
        if magic_number != validation_code:
            raise Exception("Wrong validation code")

        data = array("B", file.read())
        images = []
        for i in range(number_of_images):
            image = np.array(data[i * image_height * image_width:(i + 1) * image_height * image_width])
            image = image.reshape(-1)
            images.append(image / 256)
    return images


def load_label(url, validation_code):
    with open(url, 'rb') as file:
        magic_number, number_of_images = unpack(">II", file.read(8))
        if magic_number != validation_code:
            raise Exception("Wrong validation code")

        bdata = array("B", file.read())
        data = []
        for label in bdata:
            x = np.zeros(10)
            x[label] = 1
            data.append(x)
        data = np.array(data)
    return data


def test(img_list, label_list, number):
    plt.imshow(img_list[number], cmap=plt.cm.gray)
    plt.title(label_list[number])
    plt.show()


def get_train():
    images = load_image("data/train-images.idx3-ubyte", 2051)
    labels = load_label("data/train-labels.idx1-ubyte", 2049)
    return (images, labels)


def get_test():
    images = load_image("data/t10k-images.idx3-ubyte", 2051)
    labels = load_label("data/t10k-labels.idx1-ubyte", 2049)
    return (images, labels)


def classify(array):
    return np.array([1 if element == max(array) else 0 for element in array], dtype="float64")


if __name__ == "__main__":
    for j in range(6):
        NN = Net([784, 300, 10], False, 0.2)
        images, labels = get_train()
        # NN.train(images, labels, 60000, True)

        x, y = get_test()
        error = 0
        for i in range(len(x)):
            # print(NN(x[i]))
            # print(classify(NN(x[i])), y[i])
            if classify(NN(x[i])).tolist() != y[i].tolist():
                error += 1
        print(f"{j}:{(1 - error/len(x))*100:.2f}")
