class datasets_class:
    """
    Conjunto de parametros para cada dataset
    """

    def __init__(self) -> None:
        self.datasets = {
            # "Log likehood newton": {"path data": "../Data/",
            #                         "file data": "mnist.pkl.gz",
            #                         "path results": "../Results/",
            #                         "n": 2,
            #                         "c1": 0.1,
            #                         "c2": 0.9,
            #                         "tau": 1e-6,
            #                         "problem name": "log likehood",
            #                         "algorithm name": "newton"
            #                         },
            "Log likehood gradient": {"path data": "../Data/",
                                      "file data": "mnist.pkl.gz",
                                      "path results": "../Results/",
                                      "tau": 1e-6,
                                      "c1": 0.1,
                                      "c2": 0.9,
                                      "alpha": 1e-3,
                                      "problem name": "log likehood",
                                      "algorithm name": "descent gradient"}
        }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]
