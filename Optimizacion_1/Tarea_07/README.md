### Tarea 07 - Giovanni Gamaliel López Padilla

### Organización de los archivs

```bash
├── Results
│  ├── branin
│  │  ├── cauchy
│  │  │  ├── Iteration
│  │  │  │  ├── 01.csv
│  │  │  │  ├── ...
│  │  │  │  └── 30.csv
│  │  │  └── time_iterations.csv
│  │  ├── dogleg
│  │  │  ├── Iteration
│  │  │  │  ├── 01.csv
│  │  │  │  ├── ...
│  │  │  │  └── 30.csv
│  │  │  └── time_iterations.csv
│  │  ├── newton_cauchy
│  │  │  ├── Iteration
│  │  │  │  ├── 01.csv
│  │  │  │  ├── ...
│  │  │  │  └── 30.csv
│  │  │  └── time_iterations.csv
│  │  └── newton_modification
│  │     ├── Iteration
│  │     │  ├── 01.csv
│  │     │  ├── ...
│  │     │  └── 30.csv
│  │     └── time_iterations.csv
│  ├── rosembrock
│  │  ├── cauchy
│  │  │  ├── Iteration
│  │  │  │  ├── 01.csv
│  │  │  │  ├── ...
│  │  │  │  └── 30.csv
│  │  │  └── time_iterations.csv
│  │  ├── dogleg
│  │  │  ├── Iteration
│  │  │  │  ├── 01.csv
│  │  │  │  ├── ...
│  │  │  │  └── 30.csv
│  │  │  └── time_iterations.csv
│  │  ├── newton_cauchy
│  │  │  ├── Iteration
│  │  │  │  ├── 01.csv
│  │  │  │  ├── ...
│  │  │  │  └── 30.csv
│  │  │  └── time_iterations.csv
│  │  └── newton_modification
│  │     ├── Iteration
│  │     │  ├── 01.csv
│  │     │  ├── ...
│  │     │  └── 30.csv
│  │     └── time_iterations.csv
│  └── wood
│     ├── cauchy
│     │  ├── Iteration
│     │  │  ├── 01.csv
│     │  │  ├── ...
│     │  │  └── 30.csv
│     │  └── time_iterations.csv
│     ├── dogleg
│     │  ├── Iteration
│     │  │  ├── 01.csv
│     │  │  ├── ...
│     │  │  └── 30.csv
│     │  └── time_iterations.csv
│     ├── newton_cauchy
│     │  ├── Iteration
│     │  │  ├── 01.csv
│     │  │  ├── ...
│     │  │  └── 30.csv
│     │  └── time_iterations.csv
│     └── newton_modification
│        ├── Iteration
│        │  ├── 01.csv
│        │  ├── ...
│        │  └── 30.csv
│        └── time_iterations.csv
└── Scripts
   ├── Modules
   │  ├── functions.py
   │  ├── models.py
   │  ├── params.py
   │  └── problem.py
   ├── run.py
   ├── time_iterations_stadistics.py
   └── time_iterations_table.py
```

#### Scripts

- Modules/params.py
  Este script contiene los parametros de los metodos y funciones

- Modules/functions.py
  Contiene el calculo de las caracteristicas de cada funcion

- Modules/models.py
  Contiene la lógica de cada método y las lineas de búsqueda

- Modules/problem.py
  Lógica de la ejecucción de cada metodo dado el nombre de la función

- run.py
  Ejecucción de todas las funciones con todos los métodos

#### Output

```bash
--------------------------------------------------------------------------------
        Time
--------------------------------------------------------------------------------
Funcion       newton cauchy    newton modification      dogleg      cauchy
----------  ---------------  ---------------------  ----------  ----------
wood              1.28845               0.027438     0.330035    1.85432
rosembrock       74.3634                1.58818     41.6005     44.9421
branin            0.0171616             0.00662657   0.0159199   0.0343389
--------------------------------------------------------------------------------
        Function
--------------------------------------------------------------------------------
Funcion       newton cauchy    newton modification     dogleg      cauchy
----------  ---------------  ---------------------  ---------  ----------
wood            7.00204e-08               0.201103   2.65275   0.00010695
rosembrock      1.19599               19090.9       32.0271    0.39945
branin          0.397888                  0.648666   0.397887  0.397889
--------------------------------------------------------------------------------
        Gradient
--------------------------------------------------------------------------------
Funcion       newton cauchy    newton modification       dogleg      cauchy
----------  ---------------  ---------------------  -----------  ----------
wood            0.00190684                1.08682   0.343427     0.0198407
rosembrock      0.00176435             6112.75      1.9312       0.0430613
branin          0.000997418               0.278091  0.000877879  0.00262148
--------------------------------------------------------------------------------
                                  Iterations
--------------------------------------------------------------------------------
Funcion       newton cauchy    newton modification    dogleg    cauchy
----------  ---------------  ---------------------  --------  --------
wood                 1044                 15.4667    258.467    1538.7
rosembrock           6623.5               63.3      4912.67     6845.4
branin                 11.1                3.66667    10.2        22.5

```
