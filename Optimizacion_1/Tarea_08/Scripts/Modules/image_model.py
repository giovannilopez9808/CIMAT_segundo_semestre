from numpy import array, sum, exp, zeros
import matplotlib.pyplot as plt
from numpy.linalg import norm
from os.path import join
import cv2


class image_model:
    """
    Modelo que implementa las funciones de decision para el ploteo de las regiones y los resultados finales
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        self.epsilon = 0.01
        self.blue = 2
        self.red = 1

    def _rgb_to_class(self, rgb, n_bins):
        """
        Conversion de RGB a cada clase
        """
        x = int(rgb[0]*n_bins // 256.0)
        y = int(rgb[1]*n_bins // 256.0)
        z = int(rgb[2]*n_bins // 256.0)
        return array([x, y, z])

    def _function(self, alpha: float, mu: float, c: float, sigma: float) -> array:
        """
        funcion para cada clase
        """
        f_exp = exp(-norm(c - mu, axis=1)**2 / (2*sigma**2))
        s = alpha * f_exp
        return sum(s)

    def _get_c_label(self, c: float, alpha: array, mu: array, sigma: float) -> int:
        """
        Calculo de la probabilidad de cada histograma
        """
        alpha_1, alpha_2 = alpha
        mu_1, mu_2 = mu
        f1 = self._function(alpha_1, mu_1, c, sigma)
        f2 = self._function(alpha_2, mu_2, c, sigma)
        down = f1+f2+2*self.epsilon
        f1 = (f1+self.epsilon)/down
        f2 = (f2+self.epsilon)/down
        return self.red if f1 < f2 else self.blue

    def _get_c_labels(self, nbins: int, alpha: array, mu: array, sigma: float) -> array:
        labels = zeros((nbins, nbins, nbins), dtype=int)
        """
        Asignación del color dependiendo de la clase obtenida
        """
        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    c = array([i, j, k])
                    label = self._get_c_label(c,
                                              alpha,
                                              mu,
                                              sigma)
                    labels[i, j, k] = label
        return labels

    def segmentation(self, img: array, nbins: int, alpha: array, mu: array, sigma: float) -> array:
        """
        Segmentacion de la imagen dada
        """
        labels = self._get_c_labels(nbins,
                                    alpha,
                                    mu,
                                    sigma)
        n = img.shape[0]
        m = img.shape[1]
        self.img_seg = img.copy()
        for i in range(n):
            for j in range(m):
                c = self._rgb_to_class(img[i, j],
                                       nbins)
                label = labels[c[0], c[1], c[2]]
                self.img_seg[i, j, 0] = 255 if label == self.red else 0
                self.img_seg[i, j, 1] = 0
                self.img_seg[i, j, 2] = 255 if label == self.blue else 0

    def _H_function(self, h: array, c: array) -> float:
        """
        Obtiene el valor del histograma
        """
        h = h[c[0], c[1], c[2]]
        return h

    def _get_c_label_h(self, hist_0: array, hist_1: array, c: array) -> float:
        """
        Obtiene la clase del histograma
        """
        h1 = self._H_function(hist_0, c)
        h2 = self._H_function(hist_1, c)
        down = h1+h2 + 2.0 * self.epsilon
        h1 = (h1+self.epsilon)/down
        h2 = (h2+self.epsilon)/down
        return self.red if h1 < h2 else self.blue

    def _get_c_labels_h(self, nbins: int, hist_0: array, hist_1: array) -> array:
        """
        Asignacion a cada esspacio de la imagen
        """
        labels = zeros((nbins, nbins, nbins), dtype=int)
        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    c = array([i, j, k])
                    labels[i, j, k] = self._get_c_label_h(hist_0, hist_1, c)
        return labels

    def histogram_segmentation(self, img: array, nbins: int, hist_1: array, hist_2: array) -> array:
        """
        Segmentacion de la imagen dado el histograma
        """
        labels = self._get_c_labels_h(nbins,
                                      hist_1,
                                      hist_2)
        n = img.shape[0]
        m = img.shape[1]
        self.img_h = img.copy()
        for i in range(n):
            for j in range(m):
                c = self._rgb_to_class(img[i, j], nbins)
                label = labels[c[0]][c[1]][c[2]]
                self.img_h[i, j, 0] = 255 if label == self.red else 0
                self.img_h[i, j, 1] = 0
                self.img_h[i, j, 2] = 255 if label == self.blue else 0

    def plot(self, img_original: array, select_area: array, image_name: str) -> None:
        """
        Ploteo de los resultados
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
        ax1.set_title("Original")
        ax1.imshow(cv2.cvtColor(img_original,
                                cv2.COLOR_BGR2RGB))
        ax1.axis("off")
        ax2.set_title("Área seleccionada")
        ax2.imshow(cv2.cvtColor(select_area,
                                cv2.COLOR_BGR2RGB))
        ax2.axis("off")
        ax3.set_title("Segmentación")
        ax3.imshow(cv2.cvtColor(self.img_seg,
                                cv2.COLOR_BGR2RGB))
        ax3.axis("off")
        ax4.set_title("Histogramas")
        ax4.imshow(cv2.cvtColor(self.img_h,
                                cv2.COLOR_BGR2RGB))
        ax4.axis("off")
        plt.tight_layout()
        filename = join(self.params["path graphics"],
                        image_name,
                        "result.png")
        plt.savefig(filename)
