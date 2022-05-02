from numpy import array, loadtxt
from pandas import read_csv
from os.path import join
import cv2


class results_model:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.class_types = ["Clase_0", "Clase_1"]

    def read(self, name: str) -> None:
        self._read_image(name)
        self._read_h_files(name)
        self._read_alpha_mu(name)

    def _read_image(self, name: str) -> array:
        filename = "{}.bmp".format(name)
        filename = join(self.params["path data"],
                        filename)
        self.image = cv2.imread(filename)

    def _read_h_files(self, name: str) -> array:
        folder = join(self.params["path results"],
                      name)
        filename_h0 = join(folder,
                           "H_0.txt")
        filename_h1 = join(folder,
                           "H_1.txt")
        shape = loadtxt(filename_h0,
                        max_rows=1,
                        dtype=int)
        h_0 = loadtxt(filename_h0,
                      skiprows=1)
        self.h_0 = h_0.reshape(shape)
        h_1 = loadtxt(filename_h1,
                      skiprows=1)
        self.h_1 = h_1.reshape(shape)

    def _read_alpha_mu(self, name: str) -> tuple:
        self.alpha = ()
        self.mu = ()
        for class_type in self.class_types:
            folder = join(self.params["path results"],
                          name,
                          class_type)
            filename_alpha = join(folder,
                                  "alpha.csv")
            filename_mu = join(folder,
                               "mu.csv")
            alpha_i = read_csv(filename_alpha)
            alpha_i = alpha_i.to_numpy()
            self.alpha = self.alpha+(alpha_i,)
            mu_i = read_csv(filename_mu)
            mu_i = mu_i.to_numpy()
            self.mu = self.mu+(mu_i,)
