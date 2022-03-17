from numpy import array, delete


def mnist_image(data: array, index: int) -> array:
    vector = data[index].copy()
    vector = delete(vector, -1)
    image = vector.reshape(28, -1)
    return image
