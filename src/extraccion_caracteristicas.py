"""
extraccion_caracteristicas.py
-------------------------------------------------------------------------------
PROPOSITO:
    Este es el UNICO archivo del proyecto que usa librerias externas
    (Pillow, numpy y openpyxl). Su trabajo es muy especifico: leer el
    dataset real "Eyes-Defy-Anemia" (carpetas Italy/ e India/, cada una
    con un Excel de metadatos y subcarpetas numeradas con imagenes de
    ojo) y convertir cada caso en un puñado de NUMEROS (caracteristicas)
    que la red neuronal, hecha desde cero, pueda recibir como entrada.

    Esto NO es la red neuronal. Es el paso de "preprocesamiento /
    extraccion de caracteristicas" que describen los documentos de
    referencia: en vez de meter la imagen completa (miles de pixeles) a
    una red hecha a mano, se extraen variables numericas representativas:
        - Promedio del canal Rojo   (R)
        - Promedio del canal Verde  (G)
        - Promedio del canal Azul   (B)
        - Saturacion promedio (HSV) (S)
    y se combinan con datos estructurados del paciente (DATO NO IMAGEN):
        - Edad
        - Sexo (0 = mujer, 1 = hombre)

    La ETIQUETA (anemico=1 / no_anemico=0) NO se asigna a mano: se
    calcula automaticamente a partir del valor de hemoglobina (Hgb) de
    cada paciente, usando el criterio clinico de la OMS para adultos:
        Hgb < 13 g/dL en hombres  -> anemico
        Hgb < 12 g/dL en mujeres  -> anemico
    (Este es el mismo criterio citado en el paper de Mahmud et al.,
    "Anemia detection through non-invasive analysis of lip mucosa
    images", usado para la poblacion adulta del dataset Eyes-Defy-Anemia).

    El resultado final de este archivo es un dataset.csv con filas como:
        r_promedio, g_promedio, b_promedio, s_promedio, edad, sexo, etiqueta
    listo para que preprocesamiento.py lo normalice y red_neuronal.py
    lo use para entrenar.

ESTRUCTURA DE CARPETAS QUE ESPERA ESTE CODIGO (tal como viene el ZIP
descargado de Eyes-Defy-Anemia, SIN reorganizar nada a mano):

    data/raw_images/
    |-- Italy/
    |   |-- Italy.xlsx              <- columnas: Number, Hgb, Gender, Age, Note
    |   |-- 1/
    |   |   |-- xxxxx.jpg                       (foto original, no se usa)
    |   |   |-- xxxxx_forniceal.png             (recorte forniceal)
    |   |   |-- xxxxx_palpebral.png             (recorte palpebral) <- el que usamos
    |   |   `-- xxxxx_forniceal_palpebral.png   (recorte combinado, respaldo)
    |   |-- 2/
    |   `-- ...
    `-- India/
        |-- India.xlsx              <- mismas columnas
        |-- 1/
        |-- 2/
        `-- ...

LIBRERIAS USADAS (y por que SI estan permitidas aqui):
    - PIL (Pillow): solo para ABRIR el archivo de imagen y leer sus pixeles.
    - numpy: solo para promediar arreglos de pixeles de forma rapida.
    - openpyxl: solo para LEER el Excel de metadatos (Number, Hgb, Gender, Age).
    Ninguna de estas librerias implementa redes neuronales. La red
    neuronal en si (forward propagation, backpropagation, Adam) esta
    100% implementada a mano en red_neuronal.py usando unicamente el
    modulo "math" de Python puro.
-------------------------------------------------------------------------------
"""

import os
import csv
import struct
import io

import numpy as np
from PIL import Image
import openpyxl

from preprocesamiento import preprocesar_imagen_para_caracteristicas


# ------------------------------------------------------------------------- #
# 1. CONFIGURACION GENERAL
# ------------------------------------------------------------------------- #
CARPETA_IMAGENES = os.path.join("data", "raw_images")

# Cada pais tiene su propia subcarpeta y su propio archivo Excel.
PAISES = {
    "Italy": "Italy.xlsx",
    "India": "India.xlsx",
}

# Sufijos de archivo de imagen, en orden de preferencia.
# Se intenta primero "_palpebral" (zona mas usada para medir palidez
# segun los papers de referencia); si no existe, se usa el combinado.
SUFIJOS_IMAGEN_PREFERIDOS = [
    "_palpebral.png",
    "_forniceal_palpebral.png",
    "_forniceal.png",
]

# Edad que se usa cuando la columna "Age" del Excel tiene un valor
# invalido o esta vacia para un paciente (en vez de detener el programa).
EDAD_POR_DEFECTO = 30.0

# Umbral OMS de hemoglobina (g/dL) para adultos, segun sexo.
UMBRAL_HGB_HOMBRES = 13.0
UMBRAL_HGB_MUJERES = 12.0

# Codificacion de la columna "Gender" del Excel -> numero para la red.
# Ajustar aqui si el Excel usa otras palabras (ej. "M"/"F").
CODIGOS_GENERO = {
    "male": 1, "m": 1, "hombre": 1, "uomo": 1,
    "female": 0, "f": 0, "mujer": 0, "donna": 0,
}

# Las imagenes _palpebral.png en realidad NO tienen fondo blanco: son
# PNG con canal alfa (RGBA), donde el fondo es TRANSPARENTE (alpha=0)
# y solo la zona de la conjuntiva tiene alpha>0. Por eso la forma mas
# precisa de saber que pixel es conjuntiva real es mirar el canal alfa,
# en vez de adivinar por un umbral de "casi blanco".
#
# Si por algun motivo una imagen NO tiene canal alfa (viene en RGB
# plano con fondo blanco real), se usa este umbral como respaldo:
# un pixel se considera "fondo blanco" si sus 3 canales RGB superan
# este valor.
UMBRAL_FONDO_BLANCO = 240

TAMANO_ESTANDAR = (128, 128)


# ------------------------------------------------------------------------- #
# 2. LECTURA DEL EXCEL DE METADATOS (Number, Hgb, Gender, Age, Note)
# ------------------------------------------------------------------------- #
def convertir_a_float(valor):
    """
    Convierte un valor leido del Excel a numero decimal (float), sin
    importar si Excel lo guardo como numero real o como TEXTO con coma
    decimal en vez de punto (configuracion regional italiana/europea,
    ej. el texto "15,1" en vez del numero 15.1).

    Algunas celdas de los archivos Excel del dataset (Italy.xlsx,
    India.xlsx) pueden contener texto que NO es un numero valido por
    error de digitacion (ej. "_", "NA", "-", celdas con formulas rotas,
    etc.). En esos casos, esta funcion NO lanza una excepcion que
    detendria todo el programa: devuelve None, y quien la llama decide
    omitir esa fila (ese paciente) en vez de hacer fallar todo el
    proceso de extraccion por un solo dato sucio.
    """
    if valor is None:
        return None

    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")
        if valor == "":
            return None

    try:
        return float(valor)
    except (ValueError, TypeError):
        return None


def cargar_metadatos_excel(ruta_excel):
    """
    Lee un archivo .xlsx con columnas: Number, Hgb, Gender, Age, Note
    y devuelve un diccionario:
        { "1": {"hgb": 11.2, "genero": 0, "edad": 34.0}, ... }

    La clave del diccionario es el "Number" convertido a texto, porque
    asi se llaman las subcarpetas (carpeta "1", "2", "3", ...).
    """
    metadatos = {}

    libro = openpyxl.load_workbook(ruta_excel, data_only=True)
    hoja = libro.active

    filas = list(hoja.iter_rows(values_only=True))
    encabezados = [str(c).strip() if c is not None else "" for c in filas[0]]

    # Ubicamos el indice de cada columna por nombre, sin asumir un orden fijo.
    indice_number = encabezados.index("Number")
    indice_hgb = encabezados.index("Hgb")
    indice_gender = encabezados.index("Gender")
    indice_age = encabezados.index("Age")

    filas_omitidas_por_dato_invalido = 0

    for numero_fila_excel, fila in enumerate(filas[1:], start=2):
        if fila[indice_number] is None:
            continue  # fila vacia, se omite sin aviso (es normal al final del Excel)

        numero_paciente_float = convertir_a_float(fila[indice_number])
        if numero_paciente_float is None:
            print(f"  [AVISO] Fila {numero_fila_excel} del Excel: la columna "
                  f"'Number' tiene un valor invalido ({fila[indice_number]!r}). "
                  f"Se omite esta fila.")
            filas_omitidas_por_dato_invalido += 1
            continue
        numero_paciente = str(int(numero_paciente_float)).strip()

        valor_hgb_crudo = fila[indice_hgb]
        if valor_hgb_crudo is None or (isinstance(valor_hgb_crudo, str) and valor_hgb_crudo.strip() == ""):
            # Caso documentado: hay pacientes sin Hgb registrado (dato faltante
            # real, no un error de digitacion). Se omiten sin generar aviso
            # ruidoso, ya que es un caso esperado del dataset.
            continue

        valor_hgb = convertir_a_float(valor_hgb_crudo)
        if valor_hgb is None:
            print(f"  [AVISO] Paciente {numero_paciente} (fila {numero_fila_excel} "
                  f"del Excel): la columna 'Hgb' tiene un valor no numerico "
                  f"({valor_hgb_crudo!r}). Se omite este paciente.")
            filas_omitidas_por_dato_invalido += 1
            continue

        genero_texto = str(fila[indice_gender]).strip().lower()
        genero_numero = CODIGOS_GENERO.get(genero_texto, 0)

        edad = convertir_a_float(fila[indice_age])
        if edad is None:
            edad = EDAD_POR_DEFECTO  # dato faltante o invalido: se usa el valor por defecto

        metadatos[numero_paciente] = {
            "hgb": valor_hgb,
            "genero": genero_numero,
            "edad": edad,
        }

    if filas_omitidas_por_dato_invalido:
        print(f"  [RESUMEN] {filas_omitidas_por_dato_invalido} fila(s) omitida(s) "
              f"por datos invalidos en '{ruta_excel}'.")

    return metadatos


# ------------------------------------------------------------------------- #
# 3. CALCULO DE LA ETIQUETA (anemico / no_anemico) SEGUN CRITERIO OMS
# ------------------------------------------------------------------------- #
def calcular_etiqueta(hgb, genero_numero):
    """
    Aplica el criterio de la OMS para adultos:
        Hombre (genero_numero=1): anemico si Hgb < 13.0 g/dL
        Mujer  (genero_numero=0): anemico si Hgb < 12.0 g/dL

    Devuelve 1 si es anemico, 0 si no lo es.
    """
    umbral = UMBRAL_HGB_HOMBRES if genero_numero == 1 else UMBRAL_HGB_MUJERES
    return 1 if hgb < umbral else 0


# ------------------------------------------------------------------------- #
# 4. BUSQUEDA DE LA IMAGEN A USAR DENTRO DE UNA CARPETA DE PACIENTE
# ------------------------------------------------------------------------- #
def encontrar_imagen_palpebral(ruta_carpeta_paciente):
    """
    Dentro de la carpeta de un paciente (ej. 'data/raw_images/Italy/1/')
    busca el archivo de imagen segmentada a usar, siguiendo el orden de
    preferencia definido en SUFIJOS_IMAGEN_PREFERIDOS.

    Devuelve la ruta completa al archivo encontrado, o None si la
    carpeta no tiene ninguna de las imagenes esperadas (esto ocurre en
    los pocos casos documentados donde la conjuntiva forniceal no esta
    expuesta y solo hay 2 archivos en vez de 4).
    """
    archivos_en_carpeta = os.listdir(ruta_carpeta_paciente)

    for sufijo in SUFIJOS_IMAGEN_PREFERIDOS:
        for nombre_archivo in archivos_en_carpeta:
            if nombre_archivo.lower().endswith(sufijo):
                return os.path.join(ruta_carpeta_paciente, nombre_archivo)

    return None


# ------------------------------------------------------------------------- #
# 5. APERTURA ROBUSTA DE IMAGENES (CON REPARACION DE METADATOS PROBLEMATICOS)
# ------------------------------------------------------------------------- #
# Algunos archivos PNG del dataset Eyes-Defy-Anemia incluyen un perfil de
# color ICC embebido (chunk "iCCP") y otros metadatos de texto (zTXt,
# iTXt, etc.) que, aunque el archivo es un PNG estructuralmente valido,
# hacen que Pillow falle con el error "cannot identify image file" en
# ciertas combinaciones de version de Pillow/libpng (sobre todo en
# Windows). El contenido de la IMAGEN en si (los pixeles) esta intacto;
# el problema es exclusivamente ese metadato extra, que no se necesita
# en absoluto para calcular promedios de color.
#
# Por eso, si la apertura normal falla, se reconstruye una copia del
# archivo en memoria conservando UNICAMENTE los chunks esenciales
# (IHDR, PLTE, IDAT, IEND) y se reintenta abrir esa copia reparada.
CHUNKS_PNG_ESENCIALES = (b"IHDR", b"PLTE", b"IDAT", b"IEND")


def reparar_png_quitando_metadatos(ruta_imagen):
    """
    Lee un archivo PNG byte a byte, descarta los chunks de metadatos
    que pueden causar el error "cannot identify image file" en Pillow
    (iCCP, zTXt, iTXt, bKGD, pHYs, tIME, etc.) y devuelve los bytes de
    un PNG equivalente pero "limpio", conservando solo los chunks
    esenciales para reconstruir la imagen (IHDR, PLTE, IDAT, IEND).

    Devuelve los bytes del PNG reparado, o None si el archivo no tiene
    la firma binaria de un PNG valido (en ese caso el problema es otro
    y no se puede reparar de esta forma).
    """
    with open(ruta_imagen, "rb") as archivo:
        datos = archivo.read()

    firma_png = b"\x89PNG\r\n\x1a\n"
    if not datos.startswith(firma_png):
        return None

    posicion = len(firma_png)
    chunks_conservados = []

    while posicion < len(datos):
        if posicion + 8 > len(datos):
            break  # archivo truncado de verdad, no se puede reparar

        longitud = struct.unpack(">I", datos[posicion:posicion + 4])[0]
        tipo_chunk = datos[posicion + 4:posicion + 8]
        chunk_completo = datos[posicion:posicion + 8 + longitud + 4]

        if tipo_chunk in CHUNKS_PNG_ESENCIALES:
            chunks_conservados.append(chunk_completo)

        posicion += 8 + longitud + 4
        if tipo_chunk == b"IEND":
            break

    return firma_png + b"".join(chunks_conservados)


def abrir_imagen_robusta(ruta_imagen):
    """
    Intenta abrir una imagen normalmente con Pillow. Si falla (caso
    "cannot identify image file"), intenta repararla quitando los
    metadatos problematicos y reabrirla desde memoria.

    Lanza la excepcion original si ni siquiera la version reparada se
    puede abrir (en ese caso el archivo esta realmente corrupto, no es
    solo un problema de metadatos).
    """
    try:
        imagen = Image.open(ruta_imagen)
        imagen.load()
        return imagen
    except Exception as error_original:
        datos_reparados = reparar_png_quitando_metadatos(ruta_imagen)
        if datos_reparados is None:
            raise error_original

        imagen_reparada = Image.open(io.BytesIO(datos_reparados))
        imagen_reparada.load()
        return imagen_reparada


# ------------------------------------------------------------------------- #
# 6. CALCULO DE CARACTERISTICAS DE COLOR (IGNORANDO EL FONDO BLANCO)
# ------------------------------------------------------------------------- #
def calcular_caracteristicas_color(ruta_imagen):
    """Extrae características de color a partir de la imagen preprocesada."""
    return preprocesar_imagen_para_caracteristicas(ruta_imagen, tamaño=TAMANO_ESTANDAR)


# ------------------------------------------------------------------------- #
# 7. PROCESAMIENTO DE UN PAIS COMPLETO (Italy o India)
# ------------------------------------------------------------------------- #
def procesar_pais(nombre_pais, nombre_archivo_excel, carpeta_imagenes):
    """
    Procesa todas las subcarpetas numeradas de un pais:
        1. Lee el Excel de metadatos.
        2. Para cada numero de paciente con Hgb registrado, busca su
           imagen palpebral, calcula R/G/B/S y la etiqueta (OMS).
        3. Devuelve una lista de diccionarios, una fila por paciente.
    """
    ruta_carpeta_pais = os.path.join(carpeta_imagenes, nombre_pais)
    ruta_excel = os.path.join(ruta_carpeta_pais, nombre_archivo_excel)

    if not os.path.isdir(ruta_carpeta_pais):
        print(f"[AVISO] No existe la carpeta '{ruta_carpeta_pais}'. Se omite {nombre_pais}.")
        return []

    if not os.path.isfile(ruta_excel):
        print(f"[AVISO] No se encontro '{ruta_excel}'. Se omite {nombre_pais}.")
        return []

    metadatos = cargar_metadatos_excel(ruta_excel)
    print(f"\nPais '{nombre_pais}': {len(metadatos)} pacientes con Hgb registrado en el Excel.")

    filas_resultado = []
    omitidos_sin_imagen = 0
    omitidos_por_error = 0

    for numero_paciente, datos in metadatos.items():
        ruta_carpeta_paciente = os.path.join(ruta_carpeta_pais, numero_paciente)

        if not os.path.isdir(ruta_carpeta_paciente):
            omitidos_sin_imagen += 1
            continue

        ruta_imagen = encontrar_imagen_palpebral(ruta_carpeta_paciente)
        if ruta_imagen is None:
            omitidos_sin_imagen += 1
            continue

        try:
            r, g, b, s = calcular_caracteristicas_color(ruta_imagen)
        except Exception as error:
            print(f"  [ERROR] Paciente {numero_paciente}: {error}")
            omitidos_por_error += 1
            continue

        etiqueta = calcular_etiqueta(datos["hgb"], datos["genero"])

        filas_resultado.append({
            "r_promedio": round(r, 4),
            "g_promedio": round(g, 4),
            "b_promedio": round(b, 4),
            "s_promedio": round(s, 4),
            "edad": datos["edad"],
            "sexo": datos["genero"],
            "etiqueta": etiqueta,
        })

    print(f"  -> Procesados correctamente: {len(filas_resultado)}")
    if omitidos_sin_imagen:
        print(f"  -> Omitidos por falta de imagen: {omitidos_sin_imagen}")
    if omitidos_por_error:
        print(f"  -> Omitidos por error al leer imagen: {omitidos_por_error}")

    return filas_resultado


# ------------------------------------------------------------------------- #
# 8. CONSTRUCCION DEL DATASET FINAL (UNIENDO ITALY + INDIA)
# ------------------------------------------------------------------------- #
def construir_dataset(carpeta_imagenes=CARPETA_IMAGENES,
                       ruta_salida=os.path.join("data", "dataset.csv")):
    """
    Procesa todos los paises definidos en PAISES, junta sus resultados
    y guarda todo en data/dataset.csv.

    Columnas del dataset.csv resultante:
        r_promedio, g_promedio, b_promedio, s_promedio, edad, sexo, etiqueta
    """
    print("Iniciando extraccion de caracteristicas desde Eyes-Defy-Anemia...")

    todas_las_filas = []
    for nombre_pais, nombre_excel in PAISES.items():
        filas_pais = procesar_pais(nombre_pais, nombre_excel, carpeta_imagenes)
        todas_las_filas.extend(filas_pais)

    if not todas_las_filas:
        print("\n[ERROR] No se proceso ningun paciente. Revisa que existan las "
              "carpetas 'data/raw_images/Italy' y/o 'data/raw_images/India' "
              "con su Excel y sus subcarpetas numeradas dentro.")
        return

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    columnas = ["r_promedio", "g_promedio", "b_promedio",
                "s_promedio", "edad", "sexo", "etiqueta"]

    with open(ruta_salida, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(todas_las_filas)

    cantidad_anemicos = sum(1 for f in todas_las_filas if f["etiqueta"] == 1)
    cantidad_no_anemicos = len(todas_las_filas) - cantidad_anemicos

    print(f"\nListo. Dataset generado en '{ruta_salida}'.")
    print(f"Total de pacientes: {len(todas_las_filas)}")
    print(f"  Anemicos:     {cantidad_anemicos}")
    print(f"  No anemicos:  {cantidad_no_anemicos}")


# ------------------------------------------------------------------------- #
# 9. EJECUCION DIRECTA (PARA PROBAR ESTE ARCHIVO POR SI SOLO)
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    construir_dataset()
