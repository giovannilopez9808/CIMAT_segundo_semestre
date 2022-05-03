### Tarea 08 - Optimización

### Organizacion de archivos

```bash
├── Data
│  ├── flower.bmp
│  ├── grave.bmp
│  ├── memorial.bmp
│  ├── person1.bmp
│  ├── rose.bmp
│  └── sheep.bmp
├── Graphics
│  ├── flower
│  │  ├── result.png
│  │  └── Stokes.png
│  ├── grave
│  │  ├── result.png
│  │  └── Stokes.png
│  ├── memorial
│  │  ├── result.png
│  │  └── Stokes.png
│  ├── person1
│  │  ├── result.png
│  │  └── Stokes.png
│  ├── rose
│  │  ├── result.png
│  │  └── Stokes.png
│  └── sheep
│     ├── result.png
│     └── Stokes.png
├── Results
│  ├── flower
│  │  ├── Clase_0
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── Clase_1
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── H_0.txt
│  │  └── H_1.txt
│  ├── grave
│  │  ├── Clase_0
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── Clase_1
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── H_0.txt
│  │  └── H_1.txt
│  ├── memorial
│  │  ├── Clase_0
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── Clase_1
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── H_0.txt
│  │  └── H_1.txt
│  ├── person1
│  │  ├── Clase_0
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── Clase_1
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── H_0.txt
│  │  └── H_1.txt
│  ├── rose
│  │  ├── Clase_0
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── Clase_1
│  │  │  ├── alpha.csv
│  │  │  └── mu.csv
│  │  ├── H_0.txt
│  │  └── H_1.txt
│  └── sheep
│     ├── Clase_0
│     │  ├── alpha.csv
│     │  └── mu.csv
│     ├── Clase_1
│     │  ├── alpha.csv
│     │  └── mu.csv
│     ├── H_0.txt
│     └── H_1.txt
├── Modules
│  ├── data_model.py
│  ├── functions.py
│  ├── get_all_histograms.sh
│  ├── histograma.py
│  ├── image_model.py
│  ├── methods.py
│  ├── params.py
│  └── results_model.py
├── graphics.py
└── run.py
```

### Descripción

#### run.py

    Este programa ejecuta la optimización de los parametros para todas las imagenes

#### grapics.py

    Este programa ejecuta la creacion de las segmentaciones obtenidas por el algoritmo y los histogramas
