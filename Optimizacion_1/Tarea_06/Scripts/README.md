## Tarea 06 - Optimización

#### Organización de archivos

```bash
├── Data
│  └── mnist.pkl.gz
├── error.py
├── graphics.py
├── mnist_plot.py
├── Modules
│  ├── datasets.py
│  ├── functions.py
│  ├── methods.py
│  ├── mnist.py
│  └── plot_functions.py
├── README.md
├── Results
│  ├── beta_log_likelihood_bisection.csv
│  └── log_likelihood_bisection.csv
└── solve.py
```

#### Ejecucición

Para ejecutar la optimización de la función de log-likelihood con los datos de mnist es necesario el siguiente comando

```bash
python solve.py
```

#### Resultados

Los resultados en el modelo son guardados en la carpeta definida en el archivo `datasets.py` en el hash `path results`
