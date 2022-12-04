import cv2
import numpy as np


class ImagePolluter:
    def darken(self, image: np.ndarray, factor: float) -> np.ndarray:
        return cv2.addWeighted(image, 1 - factor, np.zeros(image.shape, image.dtype), 0, 0)

    def lighten(self, image: np.ndarray, factor: float) -> np.ndarray:
        return cv2.addWeighted(image, 1, np.zeros(image.shape, image.dtype), 0, 255 * factor)

    def blur(self, image: np.ndarray, factor: float) -> np.ndarray:
        return cv2.GaussianBlur(image, (0, 0), factor)

    def add_noise(self, image: np.ndarray, factor: float) -> np.ndarray:
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * factor
        return noisy.astype(np.uint8)

    def add_salt_pepper_noise(self, image: np.ndarray, factor: float) -> np.ndarray:
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = image.copy()
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p * factor)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p) * factor)
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out


if __name__ == "__main__":
    from os import path
    import matplotlib.pyplot as plt

    img = cv2.imread(path.join("images", "test2.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    polluter = ImagePolluter()

    fig = plt.figure(figsize=(6, 12))

    fig.add_subplot(3, 2, 1)
    plt.imshow(img)
    plt.title("Original")

    fig.add_subplot(3, 2, 2)
    plt.imshow(polluter.darken(img, 0.5))
    plt.title("Darkened")

    fig.add_subplot(3, 2, 3)
    plt.imshow(polluter.lighten(img, 0.5))
    plt.title("Lightened")

    fig.add_subplot(3, 2, 4)
    plt.imshow(polluter.blur(img, 2))
    plt.title("Blurred")

    fig.add_subplot(3, 2, 5)
    plt.imshow(polluter.add_noise(img, 1.2))
    plt.title("Noisy")

    fig.add_subplot(3, 2, 6)
    plt.imshow(polluter.add_salt_pepper_noise(img, 10))
    plt.title("Salt and Pepper")

    plt.show()
