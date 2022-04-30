from numpy import array, sum, exp, zeros
from numpy.linalg import norm


class image_model:
    def __init__(self) -> None:
        self.red = 1
        self.blue = 2

    def get_h_j(self, h: array) -> array:
        """
        c = [r, g, b]
        r \in B_1
        g \in B_2
        b \in B_3
        """
        b1, b2, b3 = h.shape
        h_j = [(h[i][j][k],
                array([i, j, k]))
               for i in range(b1)
               for j in range(b2)
               for k in range(b3)
               ]
        return h_j

    def _rgb_to_class(self, rgb, n_bins):
        x = int(rgb[0] / 256.0 * n_bins)
        y = int(rgb[1] / 256.0 * n_bins)
        z = int(rgb[2] / 256.0 * n_bins)
        return array([x, y, z])

    def _function(alpha: float, mu: float, c: float, sigma: float = 0.2) -> array:
        f_exp = exp(-norm(c - mu, axis=1)**2 / (2*sigma**2))
        s = alpha * f_exp
        return sum(s)

    def _F1(self, c: float, alpha_1: array, mu_1: array, alpha_2: array, mu_2: array, epsilon: float = 0.01, sigma: float = 0.2) -> float:
        t1 = self._function(alpha_1, mu_1, c, sigma=sigma) + epsilon
        t2 = self._function(alpha_1, mu_1, c, sigma=sigma)
        t2 += self._function(alpha_2, mu_2, c, sigma=sigma) + 2.0 * epsilon
        return t1/t2

    def _F2(self, c: float, alpha_1: array, mu_1: array, alpha_2: array, mu_2: array, epsilon: float = 0.01, sigma: float = 0.2) -> float:
        t1 = self._function(alpha_2, mu_2, c, sigma=sigma) + epsilon
        t2 = self._function(alpha_1, mu_1, c, sigma=sigma)
        t2 += self._function(alpha_2, mu_2, c, sigma=sigma) + 2.0 * epsilon
        return t1/t2

    def _get_c_label(self, c: float, alpha_1: array, mu_1: array, alpha_2: array, mu_2: array, epsilon: float = 0.01, sigma: float = 0.2) -> float:
        f1 = self._F1(c,
                      alpha_1,
                      mu_1,
                      alpha_2,
                      mu_2,
                      epsilon=epsilon,
                      sigma=sigma)
        f2 = self._F2(c,
                      alpha_1,
                      mu_1,
                      alpha_2,
                      mu_2,
                      epsilon=epsilon,
                      sigma=sigma)
        return self.red if f1 < f2 else self.blue

    def _get_c_labels(self, nbins: int, alpha_1: array, mu_1: array, alpha_2: array, mu_2: array, epsilon: float = 0.01, sigma: float = 0.2) -> array:
        labels = zeros((nbins,
                        nbins,
                        nbins),
                       dtype=int)
        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    c = array([i, j, k])
                    label = self._get_c_label(c,
                                              alpha_1,
                                              mu_1,
                                              alpha_2,
                                              mu_2,
                                              epsilon=epsilon,
                                              sigma=sigma)
                    labels[i][j][k] = label
        return labels

    def segmentation(self, img: array, nbins: int, alpha_1: array, mu_1: array, alpha_2: array, mu_2: array, epsilon: float = 0.01, sigma: float = 0.1) -> array:
        labels = self._get_c_labels(nbins,
                                    alpha_1,
                                    mu_1,
                                    alpha_2,
                                    mu_2,
                                    epsilon=epsilon,
                                    sigma=sigma)
        n, m = img.shape
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

    def _H1(self, h1: array, h2: array, c: array, epsilon: float = 0.001) -> float:
        t1 = h1[c[0]][c[1]][c[2]] + epsilon
        t2 = h1[c[0]][c[1]][c[2]] + h2[c[0]][c[1]][c[2]] + 2.0 * epsilon
        return t1/t2

    def _H2(self, h1: array, h2: array, c: array, epsilon: float = 0.001) -> float:
        t1 = h2[c[0]][c[1]][c[2]] + epsilon
        t2 = h1[c[0]][c[1]][c[2]] + h2[c[0]][c[1]][c[2]] + 2.0 * epsilon
        return t1/t2

    def _get_c_label_h(self, c: array, hist_0: array, hist_1: array) -> float:
        h1 = self._H1(hist_0, hist_1, c)
        h2 = self._H2(hist_0, hist_1, c)
        return self.blue if h1 < h2 else self.red

    def _get_c_labels_h(self, nbins: int, hist_0: array, hist_1: array) -> array:
        labels = zeros((nbins,
                        nbins,
                        nbins),
                       dtype=int)
        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    c = array([i, j, k])
                    labels[i][j][k] = self._get_c_label_h(c, hist_0, hist_1)
        return labels

    def histogram_segmentation(self, img: array, nbins: int, hist_1: array, hist_2: array) -> array:
        labels = self._get_c_labels_h(nbins,
                                      hist_1,
                                      hist_2)
        n, m = img.shape
        img_seg = img.copy()
        for i in range(n):
            for j in range(m):
                c = self._rgb_to_class(img[i][j], nbins)
                label = labels[c[0], c[1], c[2]]
                img_seg[i][j][0] = 255 if label == self.blue else 0
                img_seg[i][j][1] = 0
                img_seg[i][j][2] = 255 if label == self.red else 0
        return img_seg
