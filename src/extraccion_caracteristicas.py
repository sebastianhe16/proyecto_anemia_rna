"""
==============================================================================
 FASE 2: PROCESAMIENTO - EXTRACCIÓN DE CARACTERÍSTICAS (FEATURES)
 Proyecto: Detección de Anemia Infantil - Dataset CP-Anemic
==============================================================================

Descripción general:
---------------------
Módulo que recibe:
    - la imagen ORIGINAL en BGR (salida de cv2.imread), y
    - la MÁSCARA BINARIA de la ROI (salida de la Fase 1, 0/255)

y devuelve un vector NumPy de 12 características numéricas, calculadas
ÍNTEGRAMENTE de forma matricial/vectorizada (SIN bucles for anidados sobre
píxeles), usando EXCLUSIVAMENTE:
    - cv2   (OpenCV)   -> cv2.erode / cv2.dilate / cv2.blur / cv2.cvtColor
    - numpy             -> operaciones de álgebra matricial, indexado booleano

NO se usan librerías de Machine Learning / Deep Learning ni de extracción de
features de alto nivel (scikit-learn, tensorflow, pytorch, scipy, skimage).

Vector de salida (orden fijo, 12 posiciones):
    [0]  mediar   -> media del canal Rojo   (píxeles válidos)
    [1]  mediag   -> media del canal Verde  (píxeles válidos)
    [2]  mediab   -> media del canal Azul   (píxeles válidos)
    [3]  hhr      -> High Hue Ratio (proporción de matiz normalizado > 0.95)
    [4]  ent      -> Entropía de Shannon de la escala de grises
    [5]  b        -> Brillo medio (escala de grises, píxeles válidos)
    [6]  g1       -> media de [I(x,y) - mínimo local 3x3]
    [7]  g2       -> media de [máximo local 3x3 - I(x,y)]
    [8]  g3       -> media de [I(x,y) - media local 3x3]
    [9]  g4       -> media de [desviación estándar local 3x3]
    [10] g5       -> media de I(x,y) (nivel de gris crudo)
    [11] mean_r_g -> media de (R - G) píxel a píxel

Autor: Generado como apoyo de ingeniería de software para proyecto CP-Anemic.
==============================================================================
"""

from typing import Tuple

import cv2
import numpy as np


class ExtractorCaracteristicas:
    """
    Extractor de 12 características estadísticas/texturales sobre la
    conjuntiva palpebral, a partir de la imagen original y su máscara ROI.

    Toda la aritmética se realiza sobre matrices NumPy / operaciones de
    OpenCV, evitando bucles for pixel-a-pixel para garantizar rendimiento
    aceptable incluso en imágenes de alta resolución.
    """

    # Cantidad fija de características que produce este extractor.
    NUM_CARACTERISTICAS: int = 12

    # Nombres descriptivos, en el MISMO orden que el vector devuelto por
    # `extraer`. Útiles para construir un DataFrame o CSV en fases
    # posteriores (entrenamiento de un clasificador basado en reglas/umbrales).
    NOMBRES_CARACTERISTICAS: Tuple[str, ...] = (
        "mediar", "mediag", "mediab", "hhr", "ent", "b",
        "g1", "g2", "g3", "g4", "g5", "mean_r_g",
    )

    def __init__(self,
                 umbral_inferior: int = 20,
                 umbral_superior: int = 240,
                 umbral_hue_normalizado: float = 0.95) -> None:
        """
        Parámetros
        ----------
        umbral_inferior : int
            Valor mínimo (EXCLUSIVO) que debe tener cada canal R, G y B
            para que un píxel se considere válido (descarta sombras
            profundas / píxeles casi negros).
        umbral_superior : int
            Valor máximo (EXCLUSIVO) que debe tener cada canal R, G y B
            para que un píxel se considere válido (descarta reflejos /
            brillos saturados casi blancos).
        umbral_hue_normalizado : float
            Umbral (EXCLUSIVO, ">") sobre el matiz normalizado en [0, 1]
            para contar un píxel dentro del "High Hue Ratio" (feature 4).
        """
        self.umbral_inferior = umbral_inferior
        self.umbral_superior = umbral_superior
        self.umbral_hue_normalizado = umbral_hue_normalizado

        # Kernel 3x3 para las operaciones morfológicas de mínimo/máximo
        # local (equivalentes a un "filtro de rango" vectorizado).
        self._kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # --------------------------------------------------------------------
    # PASO 0: LIMPIEZA DE DATOS (FILTRADO ROBUSTO)
    # --------------------------------------------------------------------
    def _obtener_mascara_validos(self,
                                  r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                  mascara_roi: np.ndarray) -> np.ndarray:
        """
        Construye la máscara booleana de píxeles VÁLIDOS: deben pertenecer
        a la ROI (máscara de la Fase 1) Y tener los tres canales R, G, B
        estrictamente dentro del rango (umbral_inferior, umbral_superior).

        Esto elimina de los cálculos estadísticos posteriores:
            - Reflejos/brillos especulares (canales cercanos a 255).
            - Sombras profundas / bordes mal segmentados (canales
              cercanos a 0).

        Retorna
        -------
        np.ndarray (dtype=bool, misma forma que la imagen 2D)
            True en los píxeles que deben usarse para TODOS los cálculos
            estadísticos posteriores; False en el resto.
        """
        # 1) Máscara de pertenencia a la ROI (booleano a partir de 0/255).
        dentro_roi = mascara_roi > 0

        # 2) Condición de rango, aplicada de forma vectorizada e
        #    independiente a cada canal de color (broadcasting elemento
        #    a elemento sobre matrices 2D).
        r_valido = (r > self.umbral_inferior) & (r < self.umbral_superior)
        g_valido = (g > self.umbral_inferior) & (g < self.umbral_superior)
        b_valido = (b > self.umbral_inferior) & (b < self.umbral_superior)

        # 3) Un píxel es válido solo si cumple SIMULTÁNEAMENTE las 4
        #    condiciones (AND lógico matricial, sin bucles).
        validos = dentro_roi & r_valido & g_valido & b_valido

        return validos

    # --------------------------------------------------------------------
    # CARACTERÍSTICAS 1, 2, 3: PROMEDIOS RGB
    # --------------------------------------------------------------------
    @staticmethod
    def _calcular_promedios_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                 validos: np.ndarray) -> Tuple[float, float, float]:
        """
        Media aritmética de cada canal de color, calculada ÚNICAMENTE
        sobre los píxeles marcados como válidos (indexado booleano de
        NumPy: r[validos] devuelve un vector 1D solo con esos píxeles).
        """
        mediar = float(np.mean(r[validos]))
        mediag = float(np.mean(g[validos]))
        mediab = float(np.mean(b[validos]))
        return mediar, mediag, mediab

    # --------------------------------------------------------------------
    # CARACTERÍSTICA 4: HHR (HIGH HUE RATIO)
    # --------------------------------------------------------------------
    def _calcular_hhr(self, imagen_bgr: np.ndarray, validos: np.ndarray) -> float:
        """
        HHR = (# píxeles válidos con matiz normalizado > 0.95) / (# píxeles válidos)

        OpenCV representa el matiz (Hue) en el rango [0, 179] para
        imágenes de 8 bits (en vez de los [0, 360] grados estándar).
        Lo normalizamos a [0, 1] dividiendo por 179 antes de comparar
        contra el umbral, para trabajar en la misma escala que el
        enunciado (0.95).
        """
        # Conversión BGR -> HSV; nos quedamos únicamente con el canal H.
        imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        canal_h = imagen_hsv[:, :, 0]

        # Normalización vectorizada del matiz a [0, 1].
        h_normalizado = canal_h.astype(np.float32) / 179.0

        # Extraemos SOLO los valores de matiz de los píxeles válidos.
        h_validos = h_normalizado[validos]

        total_validos = h_validos.size
        if total_validos == 0:
            return 0.0

        # Conteo vectorizado de píxeles que superan el umbral: la
        # comparación produce un arreglo booleano, y np.sum lo interpreta
        # como 1/0 para contar cuántos True hay.
        pixeles_hue_alto = np.sum(h_validos > self.umbral_hue_normalizado)

        hhr = float(pixeles_hue_alto) / float(total_validos)
        return hhr

    # --------------------------------------------------------------------
    # CARACTERÍSTICA 5: ENTROPÍA DE SHANNON
    # --------------------------------------------------------------------
    @staticmethod
    def _calcular_entropia(gray: np.ndarray, validos: np.ndarray) -> float:
        """
        Entropía de Shannon del histograma de niveles de gris:

            H = - sum_{i=0}^{255} p_i * log2(p_i),   para p_i > 0

        donde p_i = (número de píxeles con nivel de gris i) / (total de
        píxeles válidos).

        Implementación vectorizada:
            1) np.histogram calcula, en un solo paso, las frecuencias de
               cada uno de los 256 niveles posibles (0-255) SOLO sobre
               los píxeles válidos (gray[validos]).
            2) Convertimos frecuencias absolutas a probabilidades
               dividiendo por el total de píxeles válidos.
            3) Filtramos p_i == 0 (log2(0) no está definido) usando
               indexado booleano, sin bucle explícito.
            4) Aplicamos la fórmula de Shannon con operaciones NumPy
               vectorizadas (np.log2, multiplicación elemento a elemento,
               np.sum).
        """
        valores_gray_validos = gray[validos]

        if valores_gray_validos.size == 0:
            return 0.0

        # Histograma de 256 bins (uno por cada nivel de gris posible en
        # una imagen de 8 bits), calculado solo sobre píxeles válidos.
        histograma, _ = np.histogram(
            valores_gray_validos, bins=256, range=(0, 256)
        )

        total_pixeles = valores_gray_validos.size

        # Probabilidad de ocurrencia de cada nivel de gris (vector de 256
        # posiciones que suma 1.0).
        probabilidades = histograma.astype(np.float64) / total_pixeles

        # Se descartan los niveles con probabilidad 0 (no aportan a la
        # entropía y log2(0) generaría -inf / NaN).
        probabilidades_no_nulas = probabilidades[probabilidades > 0]

        # Fórmula de Shannon, aplicada de forma vectorizada sobre todo el
        # vector de probabilidades no nulas simultáneamente.
        entropia = -np.sum(probabilidades_no_nulas * np.log2(probabilidades_no_nulas))

        return float(entropia)

    # --------------------------------------------------------------------
    # CARACTERÍSTICA 6: BRILLO
    # --------------------------------------------------------------------
    @staticmethod
    def _calcular_brillo(gray: np.ndarray, validos: np.ndarray) -> float:
        """
        Brillo medio = media aritmética de la escala de grises,
        considerando solo los píxeles válidos de la ROI.
        """
        return float(np.mean(gray[validos]))

    # --------------------------------------------------------------------
    # CARACTERÍSTICAS 7 a 11: ESTADÍSTICAS DE VECINDAD 3x3 (g1..g5)
    # --------------------------------------------------------------------
    def _calcular_estadisticas_ventana(self, gray: np.ndarray,
                                        validos: np.ndarray
                                        ) -> Tuple[float, float, float, float, float]:
        """
        Para cada píxel I(x,y), calcula estadísticas de su vecindad local
        S de 3x3 píxeles, de forma COMPLETAMENTE vectorizada (sin recorrer
        la imagen con bucles anidados):

            g1(x,y) = I(x,y) - min(S)
            g2(x,y) = max(S) - I(x,y)
            g3(x,y) = I(x,y) - mean(S)
            g4(x,y) = std(S)
            g5(x,y) = I(x,y)

        y finalmente devuelve el PROMEDIO de cada uno de esos mapas,
        calculado únicamente sobre los píxeles válidos de la ROI.

        Técnica de vectorización utilizada:
            - mínimo local 3x3  -> cv2.erode   (la erosión con un kernel
              rectangular reemplaza cada píxel por el MÍNIMO de su
              vecindad; es la operación morfológica dual del "filtro de
              mínimo").
            - máximo local 3x3  -> cv2.dilate  (la dilatación reemplaza
              cada píxel por el MÁXIMO de su vecindad).
            - media local 3x3   -> cv2.blur    (filtro de promedio /
              media aritmética de la vecindad, kernel normalizado).
            - std local 3x3     -> se deriva algebraicamente de:
                  Var(S) = E[S^2] - (E[S])^2
                  std(S) = sqrt(max(Var(S), 0))
              calculando E[S^2] con cv2.blur sobre la imagen elevada al
              cuadrado, y E[S] con cv2.blur sobre la imagen original.
              El max(..., 0) evita raíces de números negativos que
              pueden surgir por errores de redondeo en punto flotante.
        """
        # Trabajamos en float32 para evitar desbordamiento/truncamiento
        # de enteros de 8 bits al elevar al cuadrado o restar.
        gray_f = gray.astype(np.float32)

        # --- Mínimo y máximo local (operaciones morfológicas) ----------
        minimo_local = cv2.erode(gray_f, self._kernel_3x3)
        maximo_local = cv2.dilate(gray_f, self._kernel_3x3)

        # --- Media local (filtro de promedio 3x3) -----------------------
        media_local = cv2.blur(gray_f, (3, 3))

        # --- Desviación estándar local vía E[S^2] - (E[S])^2 -----------
        media_cuadrados_local = cv2.blur(gray_f ** 2, (3, 3))
        varianza_local = media_cuadrados_local - media_local ** 2
        varianza_local = np.clip(varianza_local, a_min=0.0, a_max=None)
        std_local = np.sqrt(varianza_local)

        # --- Construcción de los 5 mapas g1..g5 (matrices 2D) ----------
        mapa_g1 = gray_f - minimo_local
        mapa_g2 = maximo_local - gray_f
        mapa_g3 = gray_f - media_local
        mapa_g4 = std_local
        mapa_g5 = gray_f  # nivel de gris crudo del propio píxel

        # --- Promedio final de cada mapa, solo sobre píxeles válidos ---
        g1 = float(np.mean(mapa_g1[validos]))
        g2 = float(np.mean(mapa_g2[validos]))
        g3 = float(np.mean(mapa_g3[validos]))
        g4 = float(np.mean(mapa_g4[validos]))
        g5 = float(np.mean(mapa_g5[validos]))

        return g1, g2, g3, g4, g5

    # --------------------------------------------------------------------
    # CARACTERÍSTICA 12: DIFERENCIA DE COLOR (R - G)
    # --------------------------------------------------------------------
    @staticmethod
    def _calcular_diferencia_r_g(r: np.ndarray, g: np.ndarray,
                                  validos: np.ndarray) -> float:
        """
        Resta matricial píxel a píxel entre el canal Rojo y el canal
        Verde (R - G), y luego se promedia el resultado únicamente sobre
        los píxeles válidos de la ROI.

        Se usa float32 para permitir diferencias negativas (los canales
        originales son uint8 sin signo, donde R - G podría desbordar si
        no se convierte primero).
        """
        diferencia = r.astype(np.float32) - g.astype(np.float32)
        return float(np.mean(diferencia[validos]))

    # --------------------------------------------------------------------
    # MÉTODO PRINCIPAL: ORQUESTA TODO EL PIPELINE DE EXTRACCIÓN
    # --------------------------------------------------------------------
    def extraer(self, imagen_bgr: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
        """
        Ejecuta el pipeline completo de la Fase 2 y devuelve el vector de
        12 características.

        Parámetros
        ----------
        imagen_bgr : np.ndarray
            Imagen ORIGINAL en formato BGR (salida de cv2.imread), NO la
            ROI recortada con fondo negro -- la máscara se aplica aquí
            mismo para tener control total sobre el filtrado de píxeles.
        mascara_roi : np.ndarray
            Máscara binaria (0/255) de la conjuntiva, salida de la Fase 1
            (`segmentar_roi_conjuntiva`).

        Retorna
        -------
        np.ndarray (shape=(12,), dtype=float64)
            Vector de características en el orden documentado en
            `NOMBRES_CARACTERISTICAS`.

        Lanza
        -----
        ValueError
            Si las dimensiones de la imagen y la máscara no coinciden, o
            si no queda ningún píxel válido tras el filtrado robusto
            (imagen totalmente saturada, máscara vacía, etc.).
        """
        if imagen_bgr is None or mascara_roi is None:
            raise ValueError("La imagen y la máscara no pueden ser None.")

        if imagen_bgr.shape[:2] != mascara_roi.shape[:2]:
            raise ValueError(
                f"Las dimensiones de la imagen {imagen_bgr.shape[:2]} y la "
                f"máscara {mascara_roi.shape[:2]} no coinciden."
            )

        # --- Separación de canales de color -----------------------------
        # cv2 usa el orden BGR internamente; extraemos cada canal por
        # separado para poder aplicar los umbrales de forma independiente.
        canal_b, canal_g, canal_r = cv2.split(imagen_bgr)

        # --- Escala de grises (para entropía, brillo y ventana 3x3) -----
        gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

        # --- PASO 0: máscara de píxeles válidos --------------------------
        validos = self._obtener_mascara_validos(canal_r, canal_g, canal_b, mascara_roi)

        if not np.any(validos):
            raise ValueError(
                "No quedó ningún píxel válido tras aplicar la máscara ROI y "
                "el filtrado por rango (20, 240) en los canales R, G, B. "
                "Revise la segmentación de la Fase 1 o los umbrales usados."
            )

        # --- Características 1, 2, 3: promedios RGB ----------------------
        mediar, mediag, mediab = self._calcular_promedios_rgb(
            canal_r, canal_g, canal_b, validos
        )

        # --- Característica 4: HHR ---------------------------------------
        hhr = self._calcular_hhr(imagen_bgr, validos)

        # --- Característica 5: Entropía -----------------------------------
        ent = self._calcular_entropia(gray, validos)

        # --- Característica 6: Brillo -------------------------------------
        brillo = self._calcular_brillo(gray, validos)

        # --- Características 7-11: estadísticas de vecindad 3x3 ----------
        g1, g2, g3, g4, g5 = self._calcular_estadisticas_ventana(gray, validos)

        # --- Característica 12: diferencia R - G --------------------------
        mean_r_g = self._calcular_diferencia_r_g(canal_r, canal_g, validos)

        # --- Ensamblado del vector final, en el orden documentado --------
        vector_caracteristicas = np.array([
            mediar, mediag, mediab, hhr, ent, brillo,
            g1, g2, g3, g4, g5, mean_r_g,
        ], dtype=np.float64)

        return vector_caracteristicas


# ==============================================================================
# BLOQUE DE PRUEBA / EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    import os

    # ------------------------------------------------------------------
    # Se reutilizan las salidas típicas de la Fase 1:
    #   - imagen original
    #   - máscara binaria de la ROI (0/255)
    # Ajuste las rutas a sus archivos reales.
    # ------------------------------------------------------------------
    RUTA_IMAGEN_ORIGINAL = "resultados_fase1/muestra_001_1_original.png"
    RUTA_MASCARA_ROI = "resultados_fase1/muestra_001_4_mascara.png"

    extractor = ExtractorCaracteristicas(
        umbral_inferior=20,
        umbral_superior=240,
        umbral_hue_normalizado=0.95,
    )

    if os.path.isfile(RUTA_IMAGEN_ORIGINAL) and os.path.isfile(RUTA_MASCARA_ROI):
        # ---------- Caso real: usando salidas de la Fase 1 -------------
        imagen = cv2.imread(RUTA_IMAGEN_ORIGINAL, cv2.IMREAD_COLOR)
        mascara = cv2.imread(RUTA_MASCARA_ROI, cv2.IMREAD_GRAYSCALE)

        try:
            vector = extractor.extraer(imagen, mascara)
            print("Vector de 12 características extraído correctamente:\n")
            for nombre, valor in zip(extractor.NOMBRES_CARACTERISTICAS, vector):
                print(f"  {nombre:10s} = {valor:.4f}")
        except ValueError as error:
            print(f"[ERROR] {error}")

    else:
        # ---------- Caso demo: datos sintéticos, sin depender de la ----
        # ---------- Fase 1, solo para verificar que el módulo corre ----
        print(
            "[AVISO] No se encontraron archivos de la Fase 1 en las rutas "
            "indicadas. Ejecutando una prueba con datos sintéticos.\n"
        )

        np.random.seed(42)
        alto, ancho = 120, 160

        # Imagen sintética con tonos rojizos (simulando conjuntiva) más
        # ruido de fondo, para poder verificar el pipeline end-to-end.
        imagen_sintetica = np.random.randint(
            60, 200, size=(alto, ancho, 3), dtype=np.uint8
        )
        imagen_sintetica[:, :, 2] = np.clip(  # potenciamos el canal Rojo (BGR[2]=R)
            imagen_sintetica[:, :, 2].astype(np.int32) + 40, 0, 255
        ).astype(np.uint8)

        # Máscara sintética: una región elíptica en el centro de la imagen.
        mascara_sintetica = np.zeros((alto, ancho), dtype=np.uint8)
        cv2.ellipse(
            mascara_sintetica,
            center=(ancho // 2, alto // 2),
            axes=(ancho // 3, alto // 4),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1,
        )

        vector = extractor.extraer(imagen_sintetica, mascara_sintetica)

        print("Vector de 12 características (datos sintéticos):\n")
        for nombre, valor in zip(extractor.NOMBRES_CARACTERISTICAS, vector):
            print(f"  {nombre:10s} = {valor:.4f}")

        print(f"\nForma del vector: {vector.shape}  (esperado: (12,))")
