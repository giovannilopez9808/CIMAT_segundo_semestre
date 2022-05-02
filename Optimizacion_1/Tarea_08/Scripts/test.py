from Modules.image_model import image_model
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import loadtxt
import cv2


def plot_img(img1, img2, t='Original', s='Segmentation'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.set_title(t)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.set_title(s)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()


alpha_1 = read_csv("Results/flower/Clase_0/alpha.csv").to_numpy()
mu_1 = read_csv("Results/flower/Clase_0/mu.csv").to_numpy()
alpha_2 = read_csv("Results/flower/Clase_1/alpha.csv").to_numpy()
mu_2 = read_csv("Results/flower/Clase_1/mu.csv").to_numpy()
alpha = (alpha_1, alpha_2)
mu = (mu_1, mu_2)

# shape = loadtxt("Results/flower/H_0.txt",
#                 max_rows=1,
#                 dtype=int)
# h_0 = loadtxt("Results/flower/H_0.txt",
#               skiprows=1)
# h_0 = h_0.reshape(shape)
# h_1 = loadtxt("Results/flower/H_1.txt",
#               skiprows=1)
# h_1 = h_1.reshape(shape)
img = cv2.imread("Data/flower.bmp")
image = image_model()
img_seq = image.segmentation(img, 3, alpha, mu, 0.5)
# img_seq = image.histogram_segmentation(img, 3, h_0, h_1)
plot_img(img, img_seq)
