"""
==============================================================================
 FASE 1: PREPROCESAMIENTO Y SEGMENTACIÓN DE LA CONJUNTIVA PALPEBRAL
 Proyecto: Detección de Anemia Infantil - Dataset CP-Anemic
==============================================================================

Descripción general:
---------------------
Módulo de visión computacional "clásica" (SIN Machine Learning / Deep Learning,
sin librerías como scikit-learn, TensorFlow o PyTorch) que implementa:

    1. Carga de imágenes del dataset.
    2. Segmentación de la Región de Interés (ROI) -> conjuntiva palpebral
       inferior, mediante el Algoritmo de Umbralización Triangular aplicado
       sobre el canal a* del espacio de color CIELAB.
    3. Conversión al espacio CIELAB y extracción del canal a* (eje rojo-verde),
       tanto en formato "imagen" (para mapas de calor) como en formato
       "vector 1D" (listo para estadísticas descriptivas en la Fase 2).
    4. Visualización/guardado de: imagen original, ROI aislada y mapa de
       calor del canal a*.

Librerías utilizadas (EXCLUSIVAMENTE):
    - cv2   (OpenCV)
    - numpy
    - os

Autor: Generado como apoyo de ingeniería de software para proyecto CP-Anemic.
==============================================================================
"""

import os
import cv2
import numpy as np


# ==============================================================================
# 1. CARGA DE IMÁGENES
# ==============================================================================

def cargar_imagen(ruta_imagen: str) -> np.ndarray:
    """
    Carga una única imagen desde disco en formato BGR (estándar de OpenCV).

    Parámetros
    ----------
    ruta_imagen : str
        Ruta absoluta o relativa al archivo de imagen.

    Retorna
    -------
    np.ndarray
        Imagen en formato BGR (alto, ancho, 3).

    Lanza
    -----
    FileNotFoundError
        Si la ruta no existe en el sistema de archivos.
    ValueError
        Si OpenCV no logra decodificar el archivo (corrupto, formato
        no soportado, etc.).
    """
    if not os.path.isfile(ruta_imagen):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_imagen}")

    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError(
            f"OpenCV no pudo decodificar la imagen '{ruta_imagen}'. "
            "Verifique que el archivo no esté corrupto y que la extensión "
            "sea válida (.jpg, .png, .bmp, etc.)."
        )

    return imagen


def cargar_dataset(directorio: str,
                    extensiones=(".jpg", ".jpeg", ".png", ".bmp")) -> list:
    """
    Recorre un directorio local y carga todas las imágenes válidas del
    dataset CP-Anemic.

    Parámetros
    ----------
    directorio : str
        Carpeta donde se encuentran las imágenes del dataset.
    extensiones : tuple
        Extensiones de archivo aceptadas.

    Retorna
    -------
    list[tuple[str, np.ndarray]]
        Lista de tuplas (nombre_archivo, imagen_bgr). Las imágenes que no
        se puedan leer se omiten (con aviso en consola) en lugar de
        interrumpir todo el proceso de carga.
    """
    if not os.path.isdir(directorio):
        raise NotADirectoryError(f"El directorio no existe: {directorio}")

    dataset = []
    archivos = sorted(os.listdir(directorio))

    for nombre_archivo in archivos:
        if nombre_archivo.lower().endswith(extensiones):
            ruta_completa = os.path.join(directorio, nombre_archivo)
            try:
                imagen = cargar_imagen(ruta_completa)
                dataset.append((nombre_archivo, imagen))
            except (FileNotFoundError, ValueError) as error:
                print(f"[AVISO] Se omitió '{nombre_archivo}': {error}")

    if len(dataset) == 0:
        print(f"[AVISO] No se cargó ninguna imagen válida desde '{directorio}'.")

    return dataset


def listar_imagenes_por_clase(directorio_base: str,
                              subdirectorios=("Anemic", "Non-anemic"),
                              extensiones=(".jpg", ".jpeg", ".png", ".bmp")) -> list:
    """
    Lista las rutas de las imágenes de las carpetas de clases específicas
    dentro del dataset CP-Anemic.

    Retorna
    -------
    list[tuple[str, str]]
        Lista de tuplas (nombre_clase, ruta_imagen).
    """
    if not os.path.isdir(directorio_base):
        raise NotADirectoryError(f"El directorio base no existe: {directorio_base}")

    rutas = []
    for nombre_clase in subdirectorios:
        carpeta_clase = os.path.join(directorio_base, nombre_clase)
        if not os.path.isdir(carpeta_clase):
            print(f"[AVISO] No se encontró la carpeta '{carpeta_clase}'.")
            continue

        for nombre_archivo in sorted(os.listdir(carpeta_clase)):
            if nombre_archivo.lower().endswith(extensiones):
                rutas.append((nombre_clase, os.path.join(carpeta_clase, nombre_archivo)))

    if len(rutas) == 0:
        print(f"[AVISO] No se encontró ninguna imagen válida en '{directorio_base}'.")

    return rutas


# ==============================================================================
# 2. SEGMENTACIÓN DE LA ROI (CONJUNTIVA) - UMBRALIZACIÓN TRIANGULAR SOBRE a*
# ==============================================================================

def segmentar_roi_conjuntiva(imagen_bgr: np.ndarray, area_minima: int = 800):
    """
    Aísla la conjuntiva palpebral (zona rojiza/rosada) descartando ruido
    típico del dataset como piel, pestañas, brillos y esclerótica.

    Estrategia (matemática/matricial):
    -----------------------------------
    1) Convertimos la imagen a CIELAB y nos quedamos con el canal a*
       (eje rojo-verde). En este espacio, el rojo puro tiende a valores
       altos y el verde a valores bajos; la piel y la esclerótica quedan
       cerca del valor neutro, mientras que el tejido de la conjuntiva
       (muy vascularizado) se desplaza claramente hacia valores altos.
       Esto lo convierte en un mejor "mapa de rojez" que trabajar
       directamente sobre RGB o escala de grises.

    2) Aplicamos un suavizado Gaussiano 5x5 para atenuar ruido de alta
       frecuencia (venas finas, brillos especulares) ANTES de umbralizar,
       ya que el algoritmo Triangular es sensible a picos espurios del
       histograma.

    3) Aplicamos el Algoritmo de Umbralización Triangular
       (cv2.THRESH_TRIANGLE). Geométricamente, este método traza una
       línea recta entre el pico más alto del histograma y su extremo
       más alejado (el bin más lejano con frecuencia distinta de cero),
       y busca el punto del histograma cuya distancia perpendicular a
       esa recta es máxima; ese punto se usa como umbral. Es ideal en
       este caso porque el histograma del canal a* en fotos de conjuntiva
       suele tener UNA sola moda dominante (fondo: piel/esclerótica) con
       una "cola" hacia valores altos (conjuntiva) -- exactamente el
       escenario para el que el método Triangular fue diseñado (a
       diferencia de Otsu, pensado para histogramas bimodales
       balanceados).

    4) Limpiamos la máscara binaria resultante con operaciones
       morfológicas: apertura (elimina puntos aislados de ruido) seguida
       de cierre (rellena pequeños huecos internos, p. ej. reflejos de luz
       sobre el tejido húmedo).

    5) Buscamos contornos externos sobre la máscara limpia y seleccionamos
       el de MAYOR ÁREA, asumiendo que corresponde al párpado inferior
       (la conjuntiva suele ser la región rojiza continua más grande
       dentro del recorte del ojo). Se descarta si su área es menor a
       `area_minima`, para evitar falsos positivos (p. ej. una pequeña
       mancha de piel enrojecida).

    Parámetros
    ----------
    imagen_bgr : np.ndarray
        Imagen original en formato BGR (salida de cv2.imread).
    area_minima : int
        Área mínima (en píxeles) que debe tener el contorno encontrado
        para considerarse una ROI válida y no ruido residual.

    Retorna
    -------
    mascara_final : np.ndarray (uint8, valores 0 o 255)
        Máscara binaria de un solo canal con la forma de la conjuntiva.
    roi_bgr : np.ndarray
        Imagen original recortada por la máscara (fondo en NEGRO fuera
        de la conjuntiva).

    Lanza
    -----
    ValueError
        Si la imagen de entrada es inválida, si no se encuentra ningún
        contorno tras la umbralización, o si el contorno más grande no
        supera `area_minima`.
    """
    if imagen_bgr is None or imagen_bgr.size == 0:
        raise ValueError("La imagen de entrada está vacía o es None.")

    # --- Paso 1: pasar a CIELAB y quedarnos con el canal a* -----------------
    imagen_lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    _, canal_a, _ = cv2.split(imagen_lab)
    # Nota: OpenCV codifica a* en el rango [0, 255], donde ~128 es neutro,
    # valores > 128 tienden a rojo/magenta y valores < 128 a verde.

    # --- Paso 2: suavizado para estabilizar el histograma --------------------
    canal_a_suavizado = cv2.GaussianBlur(canal_a, (5, 5), 0)

    # --- Paso 3: Umbralización Triangular ------------------------------------
    _, mascara_bin = cv2.threshold(
        canal_a_suavizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )

    # --- Paso 4: limpieza morfológica ----------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_limpia = cv2.morphologyEx(
        mascara_bin, cv2.MORPH_OPEN, kernel, iterations=2
    )
    mascara_limpia = cv2.morphologyEx(
        mascara_limpia, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # --- Paso 5: selección del contorno principal (mayor área) --------------
    contornos, _ = cv2.findContours(
        mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        raise ValueError(
            "No se encontró ningún contorno tras la umbralización triangular. "
            "La imagen podría no contener una conjuntiva visible, tener mal "
            "enfoque, o iluminación muy pobre."
        )

    contorno_mayor = max(contornos, key=cv2.contourArea)
    area_encontrada = cv2.contourArea(contorno_mayor)

    if area_encontrada < area_minima:
        raise ValueError(
            f"El contorno más grande encontrado ({area_encontrada:.0f} px) "
            f"es menor que el área mínima permitida ({area_minima} px). "
            "Posible falso positivo o imagen de baja calidad."
        )

    # Máscara final: solo el contorno seleccionado, relleno por completo
    mascara_final = np.zeros_like(mascara_limpia)
    cv2.drawContours(mascara_final, [contorno_mayor], -1, 255, thickness=cv2.FILLED)

    # --- Extracción de la ROI a color, con fondo negro -----------------------
    roi_bgr = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara_final)

    return mascara_final, roi_bgr


# ==============================================================================
# 3. CONVERSIÓN A CIELAB Y EXTRACCIÓN DEL CANAL a*
# ==============================================================================

def extraer_canal_a_cielab(roi_bgr: np.ndarray, mascara: np.ndarray):
    """
    Convierte la ROI segmentada del espacio RGB/BGR al espacio CIELAB y
    extrae el canal a* (eje rojo-verde), que es el predictor clave de
    ausencia de palidez: valores BAJOS de a* -> poca "rojez" -> posible
    palidez/anemia; valores ALTOS -> tejido bien vascularizado ->
    coloración normal.

    Parámetros
    ----------
    roi_bgr : np.ndarray
        Imagen BGR de la ROI ya segmentada (fondo negro fuera de la
        conjuntiva), salida de `segmentar_roi_conjuntiva`.
    mascara : np.ndarray
        Máscara binaria (0/255) que indica los píxeles válidos de la
        conjuntiva, misma salida de `segmentar_roi_conjuntiva`.

    Retorna
    -------
    canal_a_visual : np.ndarray (uint8, misma forma que la máscara)
        Canal a* de la ROI en escala de grises, con 0 fuera de la
        máscara. Pensado para visualización / mapas de calor.
    valores_a_roi : np.ndarray (1D, uint8)
        Vector con ÚNICAMENTE los valores de a* de los píxeles dentro de
        la conjuntiva (sin el fondo negro), listo para calcular
        estadísticas (media, desviación estándar, mediana, percentiles,
        etc.) en la Fase 2 de extracción de características.

    Lanza
    -----
    ValueError
        Si la ROI, la máscara son None, o si la máscara no tiene ningún
        píxel activo.
    """
    if roi_bgr is None or mascara is None:
        raise ValueError(
            "La ROI o la máscara son None; ejecute primero la segmentación "
            "con `segmentar_roi_conjuntiva`."
        )

    # Conversión de espacio de color: BGR -> CIELAB
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    _, canal_a, _ = cv2.split(roi_lab)

    # Como roi_bgr ya tiene fondo negro, al convertir a LAB el canal a*
    # del fondo NO queda en 0 automáticamente (el negro BGR (0,0,0)
    # equivale a a*=128 en LAB, es decir, "neutro"). Forzamos explícitamente
    # el fondo a 0 aplicando la máscara sobre el canal a*, para que la
    # visualización distinga claramente tejido (>0) de fondo (=0).
    canal_a_visual = cv2.bitwise_and(canal_a, canal_a, mask=mascara)

    # Extraemos SOLO los píxeles pertenecientes a la conjuntiva como un
    # arreglo 1D mediante indexado booleano con la máscara. Este vector
    # es el insumo directo para funciones estadísticas de NumPy
    # (np.mean, np.std, np.median, etc.) en la siguiente fase.
    valores_a_roi = canal_a[mascara == 255]

    if valores_a_roi.size == 0:
        raise ValueError(
            "La máscara no contiene píxeles activos; no hay ROI que analizar."
        )

    return canal_a_visual, valores_a_roi


# ==============================================================================
# 4. ORQUESTADOR: PROCESAMIENTO COMPLETO DE UNA IMAGEN
# ==============================================================================

def procesar_imagen_completa(ruta_imagen: str, area_minima: int = 800) -> dict:
    """
    Ejecuta el pipeline completo de la Fase 1 sobre una sola imagen:
    carga -> segmentación triangular -> conversión CIELAB -> extracción
    del canal a*.

    Parámetros
    ----------
    ruta_imagen : str
        Ruta a la imagen del dataset CP-Anemic.
    area_minima : int
        Área mínima del contorno para considerar válida la segmentación.

    Retorna
    -------
    dict
        Diccionario con todos los resultados intermedios y finales, listo
        para pasarse a la Fase 2 (extracción de características
        estadísticas) o para visualizarse/guardarse en disco. Claves:
            - "nombre_archivo"
            - "imagen_original"   (np.ndarray BGR)
            - "mascara_roi"       (np.ndarray uint8, 0/255)
            - "roi_segmentada"    (np.ndarray BGR, fondo negro)
            - "canal_a_visual"    (np.ndarray uint8, escala de grises)
            - "mapa_calor_a"      (np.ndarray BGR, colormap JET)
            - "valores_a_roi"     (np.ndarray 1D uint8, para Fase 2)
            - "media_a"           (float)
            - "desviacion_a"      (float)
    """
    imagen_original = cargar_imagen(ruta_imagen)

    mascara, roi_bgr = segmentar_roi_conjuntiva(
        imagen_original, area_minima=area_minima
    )

    canal_a_visual, valores_a_roi = extraer_canal_a_cielab(roi_bgr, mascara)

    # Mapa de calor para inspección visual rápida de los niveles de a*.
    # cv2.COLORMAP_JET mapea valores bajos a tonos azules/fríos y valores
    # altos a tonos rojos/cálidos, lo que facilita ver de un vistazo si
    # la conjuntiva analizada luce "pálida" (predominio de azules/verdes)
    # o "sana" (predominio de rojos).
    mapa_calor = cv2.applyColorMap(canal_a_visual, cv2.COLORMAP_JET)
    # Re-aplicamos la máscara para que el fondo del mapa de calor sea negro
    # y no el color que JET asigna al valor 0.
    mapa_calor = cv2.bitwise_and(mapa_calor, mapa_calor, mask=mascara)

    resultados = {
        "nombre_archivo": os.path.basename(ruta_imagen),
        "imagen_original": imagen_original,
        "mascara_roi": mascara,
        "roi_segmentada": roi_bgr,
        "canal_a_visual": canal_a_visual,
        "mapa_calor_a": mapa_calor,
        "valores_a_roi": valores_a_roi,
        "media_a": float(np.mean(valores_a_roi)),
        "desviacion_a": float(np.std(valores_a_roi)),
    }

    return resultados


# ==============================================================================
# 5. VISUALIZACIÓN Y GUARDADO DE RESULTADOS
# ==============================================================================

def mostrar_y_guardar_resultados(resultados: dict, directorio_salida: str = None,
                                 mostrar: bool = False):
    """
    Muestra las vistas de una imagen si se solicita y, además, guarda los
    resultados en disco. Las imágenes se organizan en subcarpetas por clase
    dentro de `directorio_salida`, facilitando su posterior procesamiento.

    Parámetros
    ----------
    resultados : dict
        Salida de `procesar_imagen_completa`.
    directorio_salida : str, opcional
        Carpeta raíz donde guardar las imágenes resultantes en disco.
    mostrar : bool, opcional
        Si es True, muestra las ventanas de OpenCV para inspección visual.
    """
    nombre = resultados["nombre_archivo"]
    clase = resultados.get("clase", "sin_clase")

    print(
        f"[{clase}/{nombre}] Media canal a* en ROI: {resultados['media_a']:.2f} "
        f"| Desviación estándar: {resultados['desviacion_a']:.2f}"
    )

    if directorio_salida:
        carpeta_clase = os.path.join(directorio_salida, clase)
        os.makedirs(carpeta_clase, exist_ok=True)
        base = os.path.splitext(nombre)[0]
        cv2.imwrite(os.path.join(carpeta_clase, f"{base}_1_original.png"),
                    resultados["imagen_original"])
        cv2.imwrite(os.path.join(carpeta_clase, f"{base}_2_roi.png"),
                    resultados["roi_segmentada"])
        cv2.imwrite(os.path.join(carpeta_clase, f"{base}_3_mapa_calor_a.png"),
                    resultados["mapa_calor_a"])
        cv2.imwrite(os.path.join(carpeta_clase, f"{base}_4_mascara.png"),
                    resultados["mascara_roi"])
        print(f"[{clase}/{nombre}] Imágenes guardadas en: {carpeta_clase}")

    if mostrar:
        try:
            cv2.imshow(f"1. Original - {nombre}", resultados["imagen_original"])
            cv2.imshow(f"2. ROI Conjuntiva (fondo negro) - {nombre}", resultados["roi_segmentada"])
            cv2.imshow(f"3. Mapa de calor canal a* - {nombre}", resultados["mapa_calor_a"])
            print("Presione cualquier tecla sobre una ventana de imagen para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("[AVISO] No se pudo mostrar la ventana de visualización.")


# ==============================================================================
# BLOQUE DE PRUEBA / EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":

    DIRECTORIO_DATASET = "dataset/CP-Anemic"
    DIRECTORIO_SALIDA = "resultados_fase1"

    try:
        rutas_imagenes = listar_imagenes_por_clase(DIRECTORIO_DATASET)
    except NotADirectoryError as error:
        print(f"[ERROR] {error}")
        rutas_imagenes = []

    resultados_dataset = []

    for clase, ruta_imagen in rutas_imagenes:
        try:
            resultado = procesar_imagen_completa(ruta_imagen, area_minima=800)
            resultado["clase"] = clase
            mostrar_y_guardar_resultados(
                resultado,
                directorio_salida=DIRECTORIO_SALIDA,
                mostrar=False,
            )
            resultados_dataset.append(resultado)
        except ValueError as error:
            print(f"[AVISO] Imagen '{os.path.basename(ruta_imagen)}' descartada ({clase}): {error}")

    print(
        f"\nSe procesaron correctamente {len(resultados_dataset)} de "
        f"{len(rutas_imagenes)} imágenes del dataset."
    )
