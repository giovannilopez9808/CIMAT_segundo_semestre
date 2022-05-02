from numpy import array, sum, exp, zeros
from numpy.linalg import norm


class image_model:
    def __init__(self) -> None:
        self.red = 1
        self.blue = 2
        self.epsilon = 0.01

    def _rgb_to_class(self, rgb, n_bins):
        x = int(rgb[0]*n_bins // 256.0)
        y = int(rgb[1]*n_bins // 256.0)
        z = int(rgb[2]*n_bins // 256.0)
        return array([x, y, z])

    def _function(self, alpha: float, mu: float, c: float, sigma: float) -> array:
        f_exp = exp(-norm(c - mu, axis=1)**2 / (2*sigma**2))
        s = alpha * f_exp
        return sum(s)

    def _F(self, alpha: array, mu: array, c: float, sigma: float) -> float:
        t = self._function(alpha, mu, c, sigma)
        return t

    def _get_c_label(self, c: float, alpha: array, mu: array, sigma: float) -> float:
        alpha_1, alpha_2 = alpha
        mu_1, mu_2 = mu
        f1 = self._F(alpha_1, mu_1, c, sigma)
        f2 = self._F(alpha_2, mu_2, c, sigma)
        down = f1+f2+2*self.epsilon
        f1 = (f1+self.epsilon)/down
        f2 = (f2+self.epsilon)/down
        return self.red if f1 < f2 else self.blue

    def _get_c_labels(self, nbins: int, alpha: array, mu: array, sigma: float) -> array:
        labels = zeros((nbins,
                        nbins,
                        nbins),
                       dtype=int)
        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    c = array([i, j, k])
                    label = self._get_c_label(c,
                                              alpha,
                                              mu,
                                              sigma)
                    labels[i][j][k] = label
        return labels

    def segmentation(self, img: array, nbins: int, alpha: array, mu: array, sigma: float) -> array:
        labels = self._get_c_labels(nbins,
                                    alpha,
                                    mu,
                                    sigma=sigma)
        n = img.shape[0]
        m = img.shape[1]
        img_seg = img.copy()
        for i in range(n):
            for j in range(m):
                c = self._rgb_to_class(img[i][j],
                                       nbins)
                label = labels[c[0], c[1], c[2]]
                img_seg[i][j][0] = 255 if label == self.red else 0
                img_seg[i][j][1] = 0
                img_seg[i][j][2] = 255 if label == self.blue else 0
        return img_seg

    def _H_function(self, h: array, c: array) -> float:
        h = h[c[0], c[1], c[2]]
        return h

    def _get_c_label_h(self, hist_0: array, hist_1: array, c: array) -> float:
        h1 = self._H_function(hist_0, c)
        h2 = self._H_function(hist_1, c)
        down = h1+h2 + 2.0 * self.epsilon
        h1 = (h1+self.epsilon)/down
        h2 = (h2+self.epsilon)/down
        return self.red if h1 < h2 else self.blue

    def _get_c_labels_h(self, nbins: int, hist_0: array, hist_1: array) -> array:
        labels = zeros((nbins, nbins, nbins), dtype=int)
        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    c = array([i, j, k])
                    labels[i, j, k] = self._get_c_label_h(hist_0, hist_1, c)
        return labels

    def histogram_segmentation(self, img: array, nbins: int, hist_1: array, hist_2: array) -> array:
        labels = self._get_c_labels_h(nbins,
                                      hist_1,
                                      hist_2)
        n = img.shape[0]
        m = img.shape[1]
        img_seg = img.copy()
        for i in range(n):
            for j in range(m):
                c = self._rgb_to_class(img[i, j], nbins)
                label = labels[c[0]][c[1]][c[2]]
                img_seg[i, j, 0] = 255 if label == self.red else 0
                img_seg[i, j, 1] = 0
                img_seg[i, j, 2] = 255 if label == self.blue else 0
        return img_seg
