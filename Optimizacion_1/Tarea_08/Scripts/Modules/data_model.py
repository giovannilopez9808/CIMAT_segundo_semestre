from numpy import array, loadtxt
from os.path import join


class data_model:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.data = {"Clase 0": {"Name": "H_0.txt",
                                 "Data": []},
                     "Clase 1": {"Name": "H_1.txt",
                                 "Data": []},
                     }

    def read(self, name: str) -> array:
        path = self.params["path results"]
        for class_type in self.data:
            filename = self.data[class_type]["Name"]
            filename = join(path,
                            name,
                            filename)
            data = self._get_data(filename)
            self.data[class_type]["Data"] = data

    def _get_data(self, filename) -> array:
        shape = loadtxt(filename,
                        max_rows=1,
                        dtype=int)
        n, m, l = shape
        data = loadtxt(filename,
                       skiprows=1)
        c_list = array([[i, j, k]
                        for i in range(n)
                        for j in range(m)
                        for k in range(l)])
        self.n = n*m*l
        data = (data, c_list)
        return data

    def get_data(self, name: str) -> tuple:
        return self.data[name]["Data"]
