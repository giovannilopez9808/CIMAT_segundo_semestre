class parameters_model:
    def __init__(self) -> None:
        self.define_paths()
        self.define_graphics_parameters()

    def define_paths(self):
        self.parameters = {"path data": "../Data/",
                           "path results": "../Results/",
                           "path graphics": "../Graphics/"}

    def define_graphics_parameters(self):
        self.graphics = {"Alhóndiga": {"y lim": 35,
                                       "y delta": 5},
                         "Basilica Colegiata": {"y lim": 20,
                                                "y delta": 2},
                         "Callejón del Beso": {"y lim": 50,
                                               "y delta": 5},
                         "Casa de Diego Rivera": {"y lim": 20,
                                                  "y delta": 2},
                         "Jardín de la Unión": {"y lim": 35,
                                                "y delta": 5},
                         "Mercado Hidalgo": {"y lim": 25,
                                             "y delta": 5},
                         "Monumento Pípila": {"y lim": 60,
                                              "y delta": 5},
                         "Museo de las Momias": {"y lim": 60,
                                                 "y delta": 5},
                         "Teatro Juárez": {"y lim": 35,
                                           "y delta": 5},
                         "Universidad de Guanajuato": {"y lim": 35,
                                                       "y delta": 5}}
