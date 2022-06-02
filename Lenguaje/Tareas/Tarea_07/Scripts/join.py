from pandas import read_csv, concat

train = read_csv("train.csv")
test = read_csv("val.csv")
# labels = read_csv("labels.csv")
# test["target"] = labels["Expected"]
# test = test[test["target"] == 1]
train = concat([train, test])
train = train.reset_index()
train = train.drop(columns=["index"])
print(train)
print(train[train["target"] == 1])
train.to_csv("data2.csv", index=False)
