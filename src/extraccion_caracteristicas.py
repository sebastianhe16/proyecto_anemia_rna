"""
Extracción de características de imágenes.
Único archivo que usa PIL/numpy para leer imágenes y extraer características.
Convierte imagen RGB/HSV en características numéricas junto con edad, sexo y Hb.
"""

from PIL import Image
import numpy as np
import os


def cargar_imagen(ruta_imagen):
    """
    Carga una imagen desde la ruta especificada.
    
    Args:
        ruta_imagen (str): Ruta absoluta o relativa a la imagen
        
    Returns:
        np.ndarray: Imagen en formato RGB (altura, ancho, 3)
    """
    img = Image.open(ruta_imagen).convert('RGB')
    return np.array(img)


def extraer_estadisticas_rgb(imagen):
    """
    Extrae estadísticas de los canales RGB.
    
    Args:
        imagen (np.ndarray): Imagen en formato RGB
        
    Returns:
        dict: Diccionario con media, desviación estándar y histograma de cada canal
    """
    caracteristicas = {}
    for i, canal in enumerate(['R', 'G', 'B']):
        valores = imagen[:, :, i].flatten()
        caracteristicas[f'media_{canal}'] = np.mean(valores)
        caracteristicas[f'std_{canal}'] = np.std(valores)
        caracteristicas[f'min_{canal}'] = np.min(valores)
        caracteristicas[f'max_{canal}'] = np.max(valores)
    return caracteristicas


def extraer_caracteristicas_hsv(imagen):
    """
    Convierte a HSV y extrae características de saturación y valor.
    
    Args:
        imagen (np.ndarray): Imagen en formato RGB
        
    Returns:
        dict: Características HSV (matiz, saturación, valor)
    """
    # Normalizar a [0, 1] para conversión a HSV
    imagen_normalizada = imagen.astype(float) / 255.0
    
    # Conversión manual a HSV simplificada
    # En producción: usar cv2.cvtColor o skimage.color.rgb2hsv
    caracteristicas = {}
    
    max_vals = np.max(imagen_normalizada, axis=2)
    min_vals = np.min(imagen_normalizada, axis=2)
    
    caracteristicas['saturacion_media'] = np.mean(max_vals - min_vals)
    caracteristicas['valor_media'] = np.mean(max_vals)
    
    return caracteristicas


def extraer_todas_caracteristicas(ruta_imagen, edad=None, sexo=None, hb=None):
    """
    Extrae todas las características de una imagen incluyendo metadatos clínicos.
    
    Args:
        ruta_imagen (str): Ruta a la imagen
        edad (int, optional): Edad del paciente
        sexo (str, optional): Sexo del paciente ('M' o 'F')
        hb (float, optional): Nivel de hemoglobina
        
    Returns:
        dict: Todas las características extraídas
    """
    imagen = cargar_imagen(ruta_imagen)
    
    caracteristicas = {}
    caracteristicas.update(extraer_estadisticas_rgb(imagen))
    caracteristicas.update(extraer_caracteristicas_hsv(imagen))
    
    # Agregar metadatos clínicos si están disponibles
    if edad is not None:
        caracteristicas['edad'] = edad
    if sexo is not None:
        caracteristicas['sexo'] = sexo
    if hb is not None:
        caracteristicas['hemoglobina'] = hb
        
    return caracteristicas
