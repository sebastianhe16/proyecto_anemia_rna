"""
preprocesamiento.py
-------------------------------------------------------------------------------
PROPOSITO:
    Este archivo NO usa ninguna libreria externa (ni numpy, ni Pillow,
    ni nada fuera de la libreria estandar de Python: csv, random, json,
    math). Toma el dataset.csv generado por extraccion_caracteristicas.py
    (filas de r_promedio, g_promedio, b_promedio, s_promedio, edad,
    sexo, etiqueta) y lo deja listo para entrenar la red neuronal:

    1. NORMALIZACION MIN-MAX de cada caracteristica de entrada, usando
       la formula citada en los documentos de referencia (Doc 2 - Asare
       et al.):
                X_normalizado = (X - X_min) / (X_max - X_min)
       Esto es indispensable porque las variables de entrada estan en
       escalas muy distintas (R/G/B van de 0 a 255, la saturacion va de
       0 a 1, la edad puede ir de 0 a 90+). Sin normalizar, la red le
       daria mas "peso" e importancia a los numeros mas grandes solo
       por su magnitud, sin que eso tenga ningun sentido real.

    2. BARAJADO (shuffle) de las filas, para que el orden en que vienen
       los pacientes en el CSV (por ejemplo, todos los italianos
       primero y luego todos los indios) no afecte la division de los
       conjuntos de entrenamiento, validacion y prueba.

    3. DIVISION TRAIN / VALIDATION / TEST en proporcion 70% / 15% / 15%,
       siguiendo el mismo estandar que usa el paper de referencia
       (Fuentes-Beingolea et al., "Illumination-Robust Conjunctival
       Image Preprocessing...") sobre este mismo dataset.

    4. GUARDADO de los valores minimo y maximo usados en la
       normalizacion (parametros_normalizacion.json). Esto es CRITICO:
       cuando mas adelante se quiera clasificar un paciente nuevo, hay
       que normalizar sus caracteristicas usando estos MISMOS valores
       de minimo/maximo del entrenamiento, no unos nuevos calculados
       sobre un solo dato. Si no se guardaran estos parametros, seria
       imposible usar la red entrenada para predecir casos futuros.

ENTRADA  : data/dataset.csv
SALIDAS  : data/dataset_train.csv
           data/dataset_val.csv
           data/dataset_test.csv
           data/parametros_normalizacion.json
-------------------------------------------------------------------------------
"""

import os
import csv
import json
import random
import numpy as np
from PIL import Image


# ------------------------------------------------------------------------- #
# 1. CONFIGURACION GENERAL
# ------------------------------------------------------------------------- #
RUTA_DATASET_ENTRADA = os.path.join("data", "dataset.csv")

RUTA_SALIDA_TRAIN = os.path.join("data", "dataset_train.csv")
RUTA_SALIDA_VAL = os.path.join("data", "dataset_val.csv")
RUTA_SALIDA_TEST = os.path.join("data", "dataset_test.csv")
RUTA_PARAMETROS_NORMALIZACION = os.path.join("data", "parametros_normalizacion.json")

# Proporciones de division del dataset (deben sumar 1.0).
PROPORCION_TRAIN = 0.70
PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15

# Columnas que son CARACTERISTICAS DE ENTRADA (se normalizan).
# El orden aqui define el orden en que la red neuronal recibira los
# datos: 6 entradas, tal como indica la arquitectura "6 -> 100 -> 50 -> 1".
COLUMNAS_CARACTERISTICAS = [
    "r_promedio",
    "g_promedio",
    "b_promedio",
    "s_promedio",
    "edad",
    "sexo",
]

# Columna de salida (lo que la red debe aprender a predecir).
COLUMNA_ETIQUETA = "etiqueta"

# Semilla aleatoria fija: usar siempre el mismo numero aqui garantiza
# que, al volver a correr este script, se obtenga EXACTAMENTE la misma
# division train/val/test. Esto es importante para poder reproducir
# los resultados del informe (si se cambia el numero, el barajado y la
# division cambian).
SEMILLA_ALEATORIA = 42


# ------------------------------------------------------------------------- #
# 2. LECTURA DEL DATASET CRUDO (dataset.csv)
# ------------------------------------------------------------------------- #
def leer_dataset(ruta_csv=RUTA_DATASET_ENTRADA):
    """
    Lee data/dataset.csv y devuelve una lista de diccionarios, uno por
    paciente, con los valores ya convertidos a float/int (en el CSV
    todo se guarda como texto, hay que convertirlo de vuelta a numero).
    """
    filas = []

    with open(ruta_csv, mode="r", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        for fila_texto in lector:
            fila_numerica = {
                "r_promedio": float(fila_texto["r_promedio"]),
                "g_promedio": float(fila_texto["g_promedio"]),
                "b_promedio": float(fila_texto["b_promedio"]),
                "s_promedio": float(fila_texto["s_promedio"]),
                "edad": float(fila_texto["edad"]),
                "sexo": int(float(fila_texto["sexo"])),
                "etiqueta": int(float(fila_texto["etiqueta"])),
            }
            filas.append(fila_numerica)

    return filas


# ------------------------------------------------------------------------- #
# 3. CALCULO DE LOS PARAMETROS DE NORMALIZACION (MIN Y MAX POR COLUMNA)
# ------------------------------------------------------------------------- #
def calcular_parametros_normalizacion(filas):
    """
    Recorre todas las filas y calcula, para cada columna de
    caracteristicas, su valor minimo y maximo observado.

    IMPORTANTE: estos minimos/maximos se calculan SOLO sobre los datos
    que luego se usaran como conjunto de ENTRENAMIENTO (esto se
    garantiza en construir_conjuntos_preprocesados, que llama a esta
    funcion unicamente con la porcion de train). Calcular los
    parametros usando tambien los datos de validacion/prueba seria una
    "fuga de informacion" (data leakage): la red terminaria
    indirectamente "viendo" datos que deberian servir solo para
    evaluarla de forma honesta.

    Devuelve un diccionario:
        { "r_promedio": {"min": 10.2, "max": 230.5}, ... }
    """
    parametros = {}

    for nombre_columna in COLUMNAS_CARACTERISTICAS:
        valores_columna = [fila[nombre_columna] for fila in filas]
        parametros[nombre_columna] = {
            "min": min(valores_columna),
            "max": max(valores_columna),
        }

    return parametros


# ------------------------------------------------------------------------- #
# 4. NORMALIZACION MIN-MAX
# ------------------------------------------------------------------------- #
def normalizar_valor(valor, valor_min, valor_max):
    """
    Aplica la formula de normalizacion Min-Max:
        X_normalizado = (X - X_min) / (X_max - X_min)

    El resultado siempre queda en el rango [0, 1].

    Caso especial: si valor_max == valor_min (es decir, la columna
    tiene el MISMO valor en todos los pacientes, por ejemplo si todos
    tuvieran exactamente la misma edad), la formula normal dividiria
    entre cero. En ese caso se devuelve 0.0, ya que esa columna no
    aporta ninguna informacion para distinguir pacientes.
    """
    rango = valor_max - valor_min
    if rango == 0:
        return 0.0
    return (valor - valor_min) / rango


def normalizar_fila(fila, parametros_normalizacion):
    """
    Devuelve una NUEVA fila (diccionario) con todas las columnas de
    COLUMNAS_CARACTERISTICAS ya normalizadas al rango [0, 1], usando
    los parametros de minimo/maximo recibidos. La columna "etiqueta"
    se conserva igual (0 o 1), ya que esa es justamente lo que la red
    debe aprender a predecir, no una entrada.
    """
    fila_normalizada = {}

    for nombre_columna in COLUMNAS_CARACTERISTICAS:
        valor_original = fila[nombre_columna]
        minimo = parametros_normalizacion[nombre_columna]["min"]
        maximo = parametros_normalizacion[nombre_columna]["max"]
        fila_normalizada[nombre_columna] = normalizar_valor(valor_original, minimo, maximo)

    fila_normalizada[COLUMNA_ETIQUETA] = fila[COLUMNA_ETIQUETA]
    return fila_normalizada


def abrir_imagen_robusta(ruta_imagen):
    """Abre la imagen con respaldo simple para evitar errores de lectura."""
    imagen = Image.open(ruta_imagen)
    imagen.load()
    if imagen.mode not in {"RGB", "RGBA", "L"}:
        imagen = imagen.convert("RGBA")
    return imagen


def preprocesar_imagen_para_caracteristicas(ruta_imagen, tamaño=(128, 128)):
    """Aplica una etapa ligera de preprocesamiento basada en escala de grises y HSV."""
    imagen = abrir_imagen_robusta(ruta_imagen).resize(tamaño)
    arreglo = np.array(imagen, dtype=np.float32)

    if arreglo.ndim == 2:
        arreglo = np.stack([arreglo, arreglo, arreglo, np.ones_like(arreglo)], axis=-1)
    if arreglo.shape[-1] == 3:
        arreglo = np.dstack([arreglo, np.ones(arreglo.shape[:2], dtype=np.float32)])

    rgb = arreglo[:, :, :3] / 255.0
    alpha = arreglo[:, :, 3] if arreglo.shape[-1] > 3 else None

    if alpha is not None:
        mascara = alpha >= 180
    else:
        mascara = np.ones(rgb.shape[:2], dtype=bool)

    mascara = np.logical_and(mascara, np.any(rgb > 0.0, axis=-1))
    if not np.any(mascara):
        raise ValueError("No se encontró zona útil de la imagen para analizar")

    pixeles = rgb[mascara]
    intensidad = 0.299 * pixeles[:, 0] + 0.587 * pixeles[:, 1] + 0.114 * pixeles[:, 2]
    referencia = intensidad / (np.median(intensidad) + 1e-8)
    referencia = np.clip(referencia, 1e-4, None)

    rgb_corr = pixeles / referencia[:, None]
    rgb_corr = np.clip(rgb_corr, 0.0, 1.0)

    maximo = np.maximum(np.maximum(rgb_corr[:, 0], rgb_corr[:, 1]), rgb_corr[:, 2])
    minimo = np.minimum(np.minimum(rgb_corr[:, 0], rgb_corr[:, 1]), rgb_corr[:, 2])
    denominador = np.where(maximo == 0.0, 1.0, maximo)
    saturacion = np.where(maximo > 0.0, (maximo - minimo) / denominador, 0.0)

    r_promedio = float(np.mean(rgb_corr[:, 0]) * 255.0)
    g_promedio = float(np.mean(rgb_corr[:, 1]) * 255.0)
    b_promedio = float(np.mean(rgb_corr[:, 2]) * 255.0)
    s_promedio = float(np.mean(saturacion))
    return r_promedio, g_promedio, b_promedio, s_promedio


def normalizar_minmax(X, parametros=None):
    """Normaliza columnas con el esquema min-max y devuelve los parámetros usados."""
    X = np.array(X, dtype=float)
    if parametros is None:
        minimos = X.min(axis=0)
        maximos = X.max(axis=0)
        rango = maximos - minimos
        rango[rango == 0.0] = 1.0
        X_norm = (X - minimos) / rango
        return X_norm, minimos, maximos

    minimos = parametros["min"]
    maximos = parametros["max"]
    rango = maximos - minimos
    rango[rango == 0.0] = 1.0
    X_norm = (X - minimos) / rango
    return X_norm, minimos, maximos


def dividir_train_val_test(X, y, ratios=(0.7, 0.15, 0.15)):
    """Divide los datos en entrenamiento, validación y prueba."""
    indices = np.arange(len(y))
    np.random.seed(SEMILLA_ALEATORIA)
    np.random.shuffle(indices)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    X = X[indices]
    y = y[indices]

    n_total = len(y)
    n_train = int(round(n_total * ratios[0]))
    n_val = int(round(n_total * ratios[1]))
    n_test = n_total - n_train - n_val

    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:n_train + n_val + n_test]

    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:n_train + n_val + n_test]

    return X_train, X_val, X_test, y_train, y_val, y_test


def balancear_clases(X, y, metodo="submuestreo"):
    """Reduce el conjunto mayoritario para equilibrar las clases."""
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    indices_cero = np.where(y == 0)[0]
    indices_uno = np.where(y == 1)[0]

    if len(indices_cero) > len(indices_uno):
        indices_mayoria = indices_cero
        indices_minoria = indices_uno
    else:
        indices_mayoria = indices_uno
        indices_minoria = indices_cero

    if metodo == "submuestreo":
        indices_mayoria = np.random.choice(indices_mayoria, size=len(indices_minoria), replace=False)
    else:
        raise ValueError("Método no soportado")

    indices = np.concatenate([indices_mayoria, indices_minoria])
    np.random.shuffle(indices)
    return X[indices], y[indices]


# ------------------------------------------------------------------------- #
# 5. BARAJADO Y DIVISION TRAIN / VALIDATION / TEST
# ------------------------------------------------------------------------- #
def barajar_y_dividir(filas, proporcion_train, proporcion_val, proporcion_test,
                       semilla=SEMILLA_ALEATORIA):
    """
    Mezcla aleatoriamente el orden de las filas (para que no queden
    agrupadas por pais o por orden de carga) y las divide en 3 listas:
    entrenamiento, validacion y prueba, segun las proporciones dadas.

    Se usa una semilla fija (random.seed) para que el barajado sea
    siempre el mismo cada vez que se ejecuta este script, garantizando
    resultados reproducibles para el informe.
    """
    assert abs((proporcion_train + proporcion_val + proporcion_test) - 1.0) < 1e-9, \
        "Las proporciones de train/val/test deben sumar 1.0"

    filas_copia = list(filas)  # copia para no modificar la lista original
    random.seed(semilla)
    random.shuffle(filas_copia)

    total_filas = len(filas_copia)
    cantidad_train = int(round(total_filas * proporcion_train))
    cantidad_val = int(round(total_filas * proporcion_val))
    # El conjunto de test toma todo lo que quede, para no perder ni
    # sobrar filas por errores de redondeo.
    cantidad_test = total_filas - cantidad_train - cantidad_val

    conjunto_train = filas_copia[:cantidad_train]
    conjunto_val = filas_copia[cantidad_train:cantidad_train + cantidad_val]
    conjunto_test = filas_copia[cantidad_train + cantidad_val:]

    return conjunto_train, conjunto_val, conjunto_test


# ------------------------------------------------------------------------- #
# 6. GUARDADO DE RESULTADOS (CSVs normalizados + JSON de parametros)
# ------------------------------------------------------------------------- #
def guardar_conjunto_csv(filas, ruta_salida):
    """
    Guarda una lista de filas (diccionarios ya normalizados) en un
    archivo CSV, con las columnas en el orden de COLUMNAS_CARACTERISTICAS
    seguidas de la columna "etiqueta".
    """
    columnas = COLUMNAS_CARACTERISTICAS + [COLUMNA_ETIQUETA]

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_parametros_normalizacion(parametros, ruta_salida=RUTA_PARAMETROS_NORMALIZACION):
    """
    Guarda los minimos/maximos usados en la normalizacion en un archivo
    JSON, para poder reutilizarlos despues al clasificar pacientes
    nuevos (entrenamiento.py los genera, y main.py/red_neuronal.py los
    leera mas adelante para normalizar cualquier caso nuevo exactamente
    igual que se normalizaron los datos de entrenamiento).
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, mode="w", encoding="utf-8") as archivo:
        json.dump(parametros, archivo, indent=4, ensure_ascii=False)


# ------------------------------------------------------------------------- #
# 7. FUNCION PRINCIPAL: ORQUESTA TODO EL PREPROCESAMIENTO
# ------------------------------------------------------------------------- #
def construir_conjuntos_preprocesados(ruta_dataset=RUTA_DATASET_ENTRADA):
    """
    Ejecuta el pipeline completo de preprocesamiento:
        1. Lee dataset.csv
        2. Baraja y divide en train/val/test (70/15/15)
        3. Calcula los parametros de normalizacion SOLO con el
           conjunto de entrenamiento (para evitar fuga de informacion)
        4. Normaliza los 3 conjuntos usando esos mismos parametros
        5. Guarda los 3 CSV normalizados y el JSON de parametros
    """
    print("Iniciando preprocesamiento del dataset...")

    filas_originales = leer_dataset(ruta_dataset)
    print(f"Dataset leido: {len(filas_originales)} pacientes en total.")

    conjunto_train, conjunto_val, conjunto_test = barajar_y_dividir(
        filas_originales, PROPORCION_TRAIN, PROPORCION_VAL, PROPORCION_TEST
    )
    print(f"Division train/val/test: "
          f"{len(conjunto_train)} / {len(conjunto_val)} / {len(conjunto_test)} pacientes.")

    # Los parametros de normalizacion se calculan SOLO con el conjunto
    # de entrenamiento, para no filtrar informacion de val/test.
    parametros_normalizacion = calcular_parametros_normalizacion(conjunto_train)

    print("\nRangos [min, max] encontrados en el conjunto de entrenamiento:")
    for nombre_columna, valores in parametros_normalizacion.items():
        print(f"  {nombre_columna}: min={valores['min']:.4f}, max={valores['max']:.4f}")

    conjunto_train_normalizado = [normalizar_fila(f, parametros_normalizacion) for f in conjunto_train]
    conjunto_val_normalizado = [normalizar_fila(f, parametros_normalizacion) for f in conjunto_val]
    conjunto_test_normalizado = [normalizar_fila(f, parametros_normalizacion) for f in conjunto_test]

    guardar_conjunto_csv(conjunto_train_normalizado, RUTA_SALIDA_TRAIN)
    guardar_conjunto_csv(conjunto_val_normalizado, RUTA_SALIDA_VAL)
    guardar_conjunto_csv(conjunto_test_normalizado, RUTA_SALIDA_TEST)
    guardar_parametros_normalizacion(parametros_normalizacion)

    # Pequeño resumen de balance de clases, util para el informe.
    for nombre_conjunto, conjunto in [("train", conjunto_train),
                                       ("val", conjunto_val),
                                       ("test", conjunto_test)]:
        cantidad_anemicos = sum(1 for f in conjunto if f[COLUMNA_ETIQUETA] == 1)
        cantidad_no_anemicos = len(conjunto) - cantidad_anemicos
        print(f"\nConjunto '{nombre_conjunto}': {len(conjunto)} pacientes "
              f"({cantidad_anemicos} anemicos, {cantidad_no_anemicos} no anemicos)")

    print(f"\nListo. Archivos generados:")
    print(f"  {RUTA_SALIDA_TRAIN}")
    print(f"  {RUTA_SALIDA_VAL}")
    print(f"  {RUTA_SALIDA_TEST}")
    print(f"  {RUTA_PARAMETROS_NORMALIZACION}")


# ------------------------------------------------------------------------- #
# 8. EJECUCION DIRECTA (PARA PROBAR ESTE ARCHIVO POR SI SOLO)
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    construir_conjuntos_preprocesados()
