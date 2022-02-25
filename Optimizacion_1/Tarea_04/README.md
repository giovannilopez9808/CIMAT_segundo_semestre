# Tarea 04 - Optimización

## Giovanni Gamaliel López Padilla

### Organizacion de las carpetas

```bash
├── Results
│  ├── Problema_2
│  │  ├── Random_results
│  │  │  ├── rosembrock_2_gradient.csv
│  │  │  ├── rosembrock_2_newton.csv
│  │  │  ├── rosembrock_100_gradient.csv
│  │  │  ├── rosembrock_100_newton.csv
│  │  │  ├── wood_4_gradient.csv
│  │  │  └── wood_4_newton.csv
│  │  ├── rosembrock_2_predefined_descent_gradient.csv
│  │  ├── rosembrock_2_predefined_newton.csv
│  │  ├── rosembrock_2_random_descent_gradient.csv
│  │  ├── rosembrock_2_random_newton.csv
│  │  ├── rosembrock_100_predefined_descent_gradient.csv
│  │  ├── rosembrock_100_predefined_newton.csv
│  │  ├── rosembrock_100_random_descent_gradient.csv
│  │  ├── rosembrock_100_random_newton.csv
│  │  ├── wood_4_predefined_descent_gradient.csv
│  │  ├── wood_4_predefined_newton.csv
│  │  ├── wood_4_random_descent_gradient.csv
│  │  └── wood_4_random_newton.csv
│  └── Problema_3
│     ├── lambda_1.csv
│     ├── lambda_1_test.csv
│     ├── lambda_10.csv
│     ├── lambda_10_test.csv
│     ├── lambda_1000.csv
│     └── lambda_1000_test.csv
└── Scripts
   ├── Problema_2
   │  ├── auxiliar.py
   │  ├── datasets.py
   │  ├── functions.py
   │  ├── graphics.py
   │  ├── methods.py
   │  ├── random_solutions.py
   │  ├── stadistics_vectors.py
   │  └── unique_solutions.py
   └── Problema_3
      ├── auxiliar.py
      ├── Data
      │  └── y.txt
      ├── datasets.py
      ├── functions.py
      ├── graphic.py
      ├── methods.py
      └── solutions.py
```

### Archivos principales

Para obtener todos los resultados se debera ejecutar los programas `solutions.py` de cada carpeta.

En los programas `datasets.py` se encuentran los parametros usados para cada problema y metodo

### Ejecucion

Para realizar la ejecuccion de los programas se necesita el siguiente comando

```bash
python solutions.py
```
