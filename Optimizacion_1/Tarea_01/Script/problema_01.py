import matplotlib.pyplot as plt
import numpy as np


def f1(x, y):
    return x**2 - y**2 - 12


def f2(x, y):
    return 2 * x * y - 16


x = np.linspace(-9.9, 9.9, 1000)
sol_x1 = []
sol_x2 = []
sol_y1 = []
sol_y2 = []
for i in x:
    for j in x:
        if abs(f1(i, j)) < 1e-1:
            sol_x1.append(i)
            sol_y1.append(j)
        if abs(f2(i, j)) < 5e-2:
            sol_y2.append(j)
            sol_x2.append(i)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xticks(np.arange(-10, 12, 2))
plt.yticks(np.arange(-10, 12, 2))
plt.plot(sol_x1,
         sol_y1,
         ls="",
         marker=".",
         color="#b5179e",
         label="$f(x_1,x_2)=x^2_1-x^2_2=12$")
plt.plot(sol_x2,
         sol_y2,
         ls="",
         marker=".",
         c="#3a0ca3",
         label="$ g(x_1,x_2)=2x_1x_2=16$")
plt.legend(frameon=False, ncol=2, bbox_to_anchor=(0.87, 1.1))
plt.grid(ls="--", color="#000000", alpha=0.5)
plt.tight_layout()
plt.savefig("problema01.png", dpi=300)
