### Tarea 04 - Reconocimiento estadistico de patrones

#### Organización de las carpetas

Cada problema contiene su carpeta de graficas, resultados y data. En el caso del problema 5 la data no es añadida debido a su peso. Esta se uso con el nombre de `creditcard.csv`

```bash
├── Problema_03
│  ├── calinski_harabasz_score.py
│  ├── Graphics
│  │  ├── calinski_harabasz_score.png
│  │  ├── train_scores.png
│  │  └── validation_scores.png
│  ├── graphics_train_scores.py
│  ├── graphics_validation_scores.py
│  ├── Modules
│  │  ├── datasets.py
│  │  ├── functions.py
│  │  ├── models.py
│  │  └── normal_data.py
│  ├── Results
│  │  ├── train_data.csv
│  │  ├── train_labels.csv
│  │  ├── train_scores.csv
│  │  ├── validation_data.csv
│  │  ├── validation_labels.csv
│  │  └── validation_scores.csv
│  └── run_kmeans.py
├── Problema_04
│  ├── Data
│  │  └── Colorful-Flowers.jpg
│  ├── Graphics
│  │  ├── cluster_2.png
│  │  ├── cluster_4.png
│  │  ├── cluster_8.png
│  │  ├── cluster_16.png
│  │  ├── cluster_32.png
│  │  ├── mean_scores.png
│  │  └── minimum_scores.png
│  ├── graphics_mean_scores.py
│  ├── graphics_minimum_score.py
│  ├── Modules
│  │  ├── dataset.py
│  │  ├── functions.py
│  │  └── models.py
│  ├── Results
│  │  ├── array.csv
│  │  ├── k_means++.csv
│  │  └── random.csv
│  └── run_kmeans.py
└── Problema_05
   ├── Data
   │  └── creditcard.csv
   ├── fowlkes_mallows_score.py
   ├── Graphics
   │  ├── fowlkes_mallows_score.png
   │  └── minumum_score.png
   ├── minumum_score.py
   ├── Modules
   │  ├── dataset.py
   │  ├── functions.py
   │  └── models.py
   ├── Results
   │  ├── cluster_2.csv
   │  ├── cluster_3.csv
   │  ├── cluster_4.csv
   │  ├── cluster_5.csv
   │  ├── cluster_6.csv
   │  └── scores.csv
   └── run_kmeans.py
```

#### Archivo datasets.py

En el programa `datasets.py` se encuentra una función llamada `obtain_params` la cual regresa un diccionario con las rutas y nombres de cada archivo de datos. Si se cambia la organización presentada anteriormente esta función debera ser modificada para el funcionamiento de los programas realizados.
