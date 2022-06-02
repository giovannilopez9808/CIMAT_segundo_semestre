from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt


def obtain_distribution(data: DataFrame) -> DataFrame:
    dist = data["target"].value_counts()
    dist = 100 * dist / dist.sum()
    dist.index = ["No offensive", "Offensive"]
    return dist


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height,
                '%d' % int(height) + "%",
                ha='center',
                va='bottom')


train = read_csv("train.csv")
val = read_csv("val.csv")

train_dist = obtain_distribution(train)
val_dist = obtain_distribution(val)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
ax1.set_title("Train dataset")
ax1.set_ylim(0, 80)
bar = ax1.bar(
    train_dist.index,
    train_dist,
    color=["#99d98c", "#9d0208"],
)
autolabel(ax1, bar)
ax1.grid(ls="--", color="#000000", alpha=0.6, axis="y")
ax2.set_title("Validation dataset")
bar = ax2.bar(
    val_dist.index,
    val_dist,
    color=["#99d98c", "#9d0208"],
)
autolabel(ax2, bar)
ax2.grid(
    ls="--",
    color="#000000",
    alpha=0.6,
    axis="y",
)
plt.savefig("distribution.png", dpi=400)
