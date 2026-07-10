

from typing import Tuple

import csv
import os

import cv2
import numpy as np


class ExtractorCaracteristicas:
   

    
    NUM_CARACTERISTICAS: int = 12

    
    NOMBRES_CARACTERISTICAS: Tuple[str, ...] = (
        "mediar", "mediag", "mediab", "hhr", "ent", "b",
        "g1", "g2", "g3", "g4", "g5", "mean_r_g",
    )

    def __init__(self,
                 umbral_inferior: int = 20,
                 umbral_superior: int = 240,
                 umbral_hue_normalizado: float = 0.95) -> None:
        
        self.umbral_inferior = umbral_inferior
        self.umbral_superior = umbral_superior
        self.umbral_hue_normalizado = umbral_hue_normalizado

        
        self._kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    
    def _obtener_mascara_validos(self,
                                  r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                  mascara_roi: np.ndarray) -> np.ndarray:
        
        dentro_roi = mascara_roi > 0

        
        r_valido = (r > self.umbral_inferior) & (r < self.umbral_superior)
        g_valido = (g > self.umbral_inferior) & (g < self.umbral_superior)
        b_valido = (b > self.umbral_inferior) & (b < self.umbral_superior)

        
        validos = dentro_roi & r_valido & g_valido & b_valido

        return validos

    
    @staticmethod
    def _calcular_promedios_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                 validos: np.ndarray) -> Tuple[float, float, float]:
        
        mediar = float(np.mean(r[validos]))
        mediag = float(np.mean(g[validos]))
        mediab = float(np.mean(b[validos]))
        return mediar, mediag, mediab

    
    def _calcular_hhr(self, imagen_bgr: np.ndarray, validos: np.ndarray) -> float:
        
        
        imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        canal_h = imagen_hsv[:, :, 0]

       
        h_normalizado = canal_h.astype(np.float32) / 179.0

        
        h_validos = h_normalizado[validos]

        total_validos = h_validos.size
        if total_validos == 0:
            return 0.0

       
        pixeles_hue_alto = np.sum(h_validos > self.umbral_hue_normalizado)

        hhr = float(pixeles_hue_alto) / float(total_validos)
        return hhr

    
    @staticmethod
    def _calcular_entropia(gray: np.ndarray, validos: np.ndarray) -> float:
        
        valores_gray_validos = gray[validos]

        if valores_gray_validos.size == 0:
            return 0.0

        
        histograma, _ = np.histogram(
            valores_gray_validos, bins=256, range=(0, 256)
        )

        total_pixeles = valores_gray_validos.size

        
        probabilidades = histograma.astype(np.float64) / total_pixeles

        
        probabilidades_no_nulas = probabilidades[probabilidades > 0]

        
        entropia = -np.sum(probabilidades_no_nulas * np.log2(probabilidades_no_nulas))

        return float(entropia)

    
    @staticmethod
    def _calcular_brillo(gray: np.ndarray, validos: np.ndarray) -> float:
        
        return float(np.mean(gray[validos]))


    def _calcular_estadisticas_ventana(self, gray: np.ndarray,
                                        validos: np.ndarray
                                        ) -> Tuple[float, float, float, float, float]:
        
        gray_f = gray.astype(np.float32)

        
        minimo_local = cv2.erode(gray_f, self._kernel_3x3)
        maximo_local = cv2.dilate(gray_f, self._kernel_3x3)

        
        media_local = cv2.blur(gray_f, (3, 3))

       
        media_cuadrados_local = cv2.blur(gray_f ** 2, (3, 3))
        varianza_local = media_cuadrados_local - media_local ** 2
        varianza_local = np.clip(varianza_local, a_min=0.0, a_max=None)
        std_local = np.sqrt(varianza_local)

        
        mapa_g1 = gray_f - minimo_local
        mapa_g2 = maximo_local - gray_f
        mapa_g3 = gray_f - media_local
        mapa_g4 = std_local
        mapa_g5 = gray_f  

        
        g1 = float(np.mean(mapa_g1[validos]))
        g2 = float(np.mean(mapa_g2[validos]))
        g3 = float(np.mean(mapa_g3[validos]))
        g4 = float(np.mean(mapa_g4[validos]))
        g5 = float(np.mean(mapa_g5[validos]))

        return g1, g2, g3, g4, g5

    
    @staticmethod
    def _calcular_diferencia_r_g(r: np.ndarray, g: np.ndarray,
                                  validos: np.ndarray) -> float:
        
        diferencia = r.astype(np.float32) - g.astype(np.float32)
        return float(np.mean(diferencia[validos]))

   
    def extraer(self, imagen_bgr: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
        
        if imagen_bgr is None or mascara_roi is None:
            raise ValueError("La imagen y la máscara no pueden ser None.")

        if imagen_bgr.shape[:2] != mascara_roi.shape[:2]:
            raise ValueError(
                f"Las dimensiones de la imagen {imagen_bgr.shape[:2]} y la "
                f"máscara {mascara_roi.shape[:2]} no coinciden."
            )

       
        canal_b, canal_g, canal_r = cv2.split(imagen_bgr)

       
        gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

        
        validos = self._obtener_mascara_validos(canal_r, canal_g, canal_b, mascara_roi)

        if not np.any(validos):
            raise ValueError(
                "No quedó ningún píxel válido tras aplicar la máscara ROI y "
                "el filtrado por rango (20, 240) en los canales R, G, B. "
                "Revise la segmentación de la Fase 1 o los umbrales usados."
            )

        
        mediar, mediag, mediab = self._calcular_promedios_rgb(
            canal_r, canal_g, canal_b, validos
        )

        
        hhr = self._calcular_hhr(imagen_bgr, validos)

        
        ent = self._calcular_entropia(gray, validos)

        
        brillo = self._calcular_brillo(gray, validos)

        
        g1, g2, g3, g4, g5 = self._calcular_estadisticas_ventana(gray, validos)

        
        mean_r_g = self._calcular_diferencia_r_g(canal_r, canal_g, validos)

       
        vector_caracteristicas = np.array([
            mediar, mediag, mediab, hhr, ent, brillo,
            g1, g2, g3, g4, g5, mean_r_g,
        ], dtype=np.float64)

        return vector_caracteristicas




def listar_pares_original_mascara(directorio_base: str,
                                  subdirectorios=("Anemic", "Non-anemic")) -> list:
    
    pares = []

    for nombre_clase in subdirectorios:
        carpeta_clase = os.path.join(directorio_base, nombre_clase)
        if not os.path.isdir(carpeta_clase):
            print(f"[AVISO] No se encontró la carpeta '{carpeta_clase}'.")
            continue

        for nombre_archivo in sorted(os.listdir(carpeta_clase)):
            if not nombre_archivo.endswith("_1_original.png"):
                continue

            prefijo = nombre_archivo[:-len("_1_original.png")]
            ruta_mascara = os.path.join(carpeta_clase, f"{prefijo}_4_mascara.png")
            if not os.path.isfile(ruta_mascara):
                print(
                    f"[AVISO] Se omitió '{nombre_archivo}' porque no existe "
                    f"la máscara correspondiente: {ruta_mascara}"
                )
                continue

            pares.append({
                "clase": nombre_clase,
                "prefijo": prefijo,
                "ruta_original": os.path.join(carpeta_clase, nombre_archivo),
                "ruta_mascara": ruta_mascara,
            })

    return pares


def guardar_csv_caracteristicas(directorio_resultados_fase1: str,
                                ruta_csv: str,
                                subdirectorios=("Anemic", "Non-anemic")) -> int:
    
    extractor = ExtractorCaracteristicas(
        umbral_inferior=20,
        umbral_superior=240,
        umbral_hue_normalizado=0.95,
    )

    pares = listar_pares_original_mascara(directorio_resultados_fase1, subdirectorios)

    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)

    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(
            ["clase", "IMAGE_ID", "ruta_original", "ruta_mascara", *extractor.NOMBRES_CARACTERISTICAS]
        )

        procesadas = 0
        for item in pares:
            imagen = cv2.imread(item["ruta_original"], cv2.IMREAD_COLOR)
            mascara = cv2.imread(item["ruta_mascara"], cv2.IMREAD_GRAYSCALE)

            if imagen is None or mascara is None:
                print(f"[AVISO] No se pudo leer '{item['ruta_original']}'.")
                continue

            try:
                vector = extractor.extraer(imagen, mascara)
            except ValueError as error:
                print(f"[AVISO] No se pudo procesar {item['prefijo']}: {error}")
                continue

            writer.writerow([
                item["clase"],
                item["prefijo"],
                item["ruta_original"],
                item["ruta_mascara"],
                *vector.tolist(),
            ])
            procesadas += 1
            print(f"[OK] {item['clase']}/{item['prefijo']} -> características extraídas")

    print(f"\nSe guardaron {procesadas} registros en: {ruta_csv}")
    return procesadas




if __name__ == "__main__":
    DIRECTORIO_RESULTADOS = "resultados_fase1"
    RUTA_CSV = os.path.join(DIRECTORIO_RESULTADOS, "caracteristicas_extraidas.csv")
    guardar_csv_caracteristicas(DIRECTORIO_RESULTADOS, RUTA_CSV)

