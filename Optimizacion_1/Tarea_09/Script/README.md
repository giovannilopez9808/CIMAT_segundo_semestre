## Tarea 10 - Optimización

### Organización de archivos

```bash
├── Data
│  └── lenanoise.png
├── Modules
│  ├── functions.py
│  ├── GC_methods.py
│  ├── methods.py
│  └── params.py
├── Graphics_function_gradient.py
├── graphics.py
└── run.py
```

### Ejeccucion del programa

El optimizador se ejecuta con el programa llamado `run.py`. Dentro de este programa se encuentra una variable de tipo diccionario con la siguiente estructura

```python
datasets = {
    "lambda": lambda_value,
    "tau": tau,
    "GC method": method
}
```

donde

- `method`: nombre de cada tamaño de paso para $\beta_k$. Los posibles valores son `"FR", "PR", "HS", "FR PR"`.

- `lambda_value`: Es el valor del parámetro de $\lambda$.

- `tau`: Valor mínimo para realizar la detención del algoritmo.
