

import os
import cv2
import numpy as np




def cargar_imagen(ruta_imagen: str) -> np.ndarray:
    
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




def segmentar_roi_conjuntiva(imagen_bgr: np.ndarray, area_minima: int = 800):
    
    if imagen_bgr is None or imagen_bgr.size == 0:
        raise ValueError("La imagen de entrada está vacía o es None.")

    
    imagen_lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    _, canal_a, _ = cv2.split(imagen_lab)

    canal_a_suavizado = cv2.GaussianBlur(canal_a, (5, 5), 0)


    _, mascara_bin = cv2.threshold(
        canal_a_suavizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_limpia = cv2.morphologyEx(
        mascara_bin, cv2.MORPH_OPEN, kernel, iterations=2
    )
    mascara_limpia = cv2.morphologyEx(
        mascara_limpia, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    
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

    
    mascara_final = np.zeros_like(mascara_limpia)
    cv2.drawContours(mascara_final, [contorno_mayor], -1, 255, thickness=cv2.FILLED)


    roi_bgr = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara_final)

    return mascara_final, roi_bgr




def extraer_canal_a_cielab(roi_bgr: np.ndarray, mascara: np.ndarray):
    
    if roi_bgr is None or mascara is None:
        raise ValueError(
            "La ROI o la máscara son None; ejecute primero la segmentación "
            "con `segmentar_roi_conjuntiva`."
        )

    
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    _, canal_a, _ = cv2.split(roi_lab)

    
    canal_a_visual = cv2.bitwise_and(canal_a, canal_a, mask=mascara)

    
    valores_a_roi = canal_a[mascara == 255]

    if valores_a_roi.size == 0:
        raise ValueError(
            "La máscara no contiene píxeles activos; no hay ROI que analizar."
        )

    return canal_a_visual, valores_a_roi




def procesar_imagen_completa(ruta_imagen: str, area_minima: int = 800) -> dict:
    
    imagen_original = cargar_imagen(ruta_imagen)

    mascara, roi_bgr = segmentar_roi_conjuntiva(
        imagen_original, area_minima=area_minima
    )

    canal_a_visual, valores_a_roi = extraer_canal_a_cielab(roi_bgr, mascara)

    mapa_calor = cv2.applyColorMap(canal_a_visual, cv2.COLORMAP_JET)

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




def mostrar_y_guardar_resultados(resultados: dict, directorio_salida: str = None,
                                 mostrar: bool = False):
    
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
