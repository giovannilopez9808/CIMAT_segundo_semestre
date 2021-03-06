import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parameters = {"path results": "../Output/",
              "path graphics": "../../Document/Graphics/",
              "graphics file": "boxplot.png",
              "channel 34 file": "channel_34.dat",
              "channel 39 file": "channel_39.dat",
              "channel 49 file": "channel_49.dat"}

c_34 = np.loadtxt("{}{}".format(parameters["path results"],
                                parameters["channel 34 file"]))
c_39 = np.loadtxt("{}{}".format(parameters["path results"],
                                parameters["channel 39 file"]))
c_49 = np.loadtxt("{}{}".format(parameters["path results"],
                                parameters["channel 49 file"]))
data = {"34 canales": c_34,
        "39 canales": c_39,
        "49 canales": c_49}
data = pd.DataFrame(data)
print(data.describe().transpose())
plt.subplots(figsize=(6, 4))
plt.boxplot(data)
plt.xticks([1, 2, 3],
           data.columns)
plt.ylim(4e9, 9e9)
plt.tight_layout()
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["graphics file"]))
