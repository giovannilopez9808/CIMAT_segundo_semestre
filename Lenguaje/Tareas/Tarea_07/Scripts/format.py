from pandas import read_csv, concat

from re import sub


def remove_quotes(text: str) -> str:
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return text


train = read_csv("hateval2019_es_train.csv")
dev = read_csv("hateval2019_es_dev.csv")
test = read_csv("hateval2019_es_test.csv")

data = concat([train, dev, test])
data = data.reset_index()
data = data.drop(columns=["index", "id", "HS", "TR"])
data.columns = ["text", "target"]
# data = data[data["target"] == 1]
data["text"] = data["text"].apply(lambda x: sub(r"@(\w){1,15}", "@USUARIO", x))
data["text"] = data["text"].apply(lambda x: sub(r'http\S+', "<URL>", x))
# data["text"] = data["text"].apply(lambda x: x.replace(",", ""))
# train = read_csv("train.csv")
# data = concat([data, train])
data["text"] = data["text"].apply(lambda x: x.replace('"', ""))
data["text"] = data["text"].apply(lambda x: x.replace(",", ""))
print(len(data[data["target"] == 1]) / len(data))
data.to_csv("val_2.csv", index=False)


train = read_csv("train.csv")
val = read_csv("val.csv")
val = val[val["target"] == 1]
# test = read_csv("test.csv")
# labels = read_csv("labels.csv")
# test["target"] = labels["Expected"]
data = concat([train, val, val])
data = data.reset_index()
data = data.drop(columns=["index"])
data.to_csv("train_2.csv",
            index=False)
