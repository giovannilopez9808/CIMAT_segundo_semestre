## Tarea 02

### Giovanni Gamaliel López Padilla

### Procesamiento de Lenguaje Natural

Esta tarea fue organizada en diferentes archivos debido a que en su desarrollo el kernel de python se rompia durante su ejección. Provocando así que no se obtuvieras resultados de cada punto.

#### Uso

Cada archivo contiene una función llamada `obtain_parameters`. Esta función regresa un diccionario que contiene la ruta y nombre de los archivos que se usaron en la tarea. Su organización y valores son las siguientes:

```python
parameters = {
        # Ruta de los archivos
        "path data": "../Data/",
        # Archivos de entrenamiento
        "train": {
            "data": "mex_train.txt",
            "labels": "mex_train_labels.txt"
        },
        # Archivos de validacion
        "validation": {
            "data": "mex_val.txt",
            "labels": "mex_val_labels.txt"
        },
        # Archivo de EmoLex
        "EmoLex": "emolex.txt",
        # Archivo de SEL
        "SEL": "SEL.txt",
        "max words": 5000,
        "max bigrams": 1000,
    }
```

Para modificar estos valores se puede usar de la siguiente manera: Supongamos que los datos se encuentran en la misma posición que los notebooks. Entonces, se deberá de añadir la siguiente linea:

```python
parameters = obtain_parameters()
parameters["path"] = "/"
```

#### Organización

Los puntos que contiene cada notebooks son los siguientes:

- Tarea02_01.ipynb
  Puntos 2.1 al 2.7

- Tarea02_02.ipynb
  Puntos 2.8

- Tarea02_03.ipynb
  Puntos 2.9, 2.10, Puntos 3 al 3.2

- Tarea02_04.ipynb
  Punto 4.1

- Tarea02_05.ipynb
  Puntos 4.2 y 4.3

#### Archivo functios.py

En el archivo `functions.py` se encuentran las funciones creadas para todos los notebooks, junto con las librerias utilizadas. Se recurrio a esto debido a la partición de los notebooks.
