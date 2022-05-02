from Modules.image_model import image_model
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import loadtxt
import cv2


def plot_img(img1, img2, t='Original', s='Segmentation'):
    fig = plt.figure(figsize=(8, 8))
    rows = 1
    columns = 2
    ax1 = fig.add_subplot(rows, columns, 1)
    ax1.set_title(t)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(rows, columns, 2)
    ax2.set_title(s)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()


alpha_1 = read_csv("Results/flower/Clase_0/alpha.csv").to_numpy()
mu_1 = read_csv("Results/flower/Clase_0/mu.csv").to_numpy()
alpha_2 = read_csv("Results/flower/Clase_1/alpha.csv").to_numpy()
mu_2 = read_csv("Results/flower/Clase_1/mu.csv").to_numpy()
shape = loadtxt("Results/flower/H_0.txt",
                max_rows=1,
                dtype=int)
h_0 = loadtxt("Results/flower/H_0.txt",
              skiprows=1)
h_0 = h_0.reshape(shape)
h_1 = loadtxt("Results/flower/H_0.txt",
              skiprows=1)
h_1 = h_1.reshape(shape)
alpha = (alpha_1, alpha_2)
mu = (mu_1, mu_2)
img = cv2.imread("Data/flower.bmp")
image = image_model()
# img_seg = image.segmentation(img, 3, alpha, mu, 0.01, 10)
img_seq = image.histogram_segmentation(img, 3, h_0, h_1)
plot_img(img, img_seq)
