"""
Utilidades generales del proyecto.
Incluye guardar/cargar pesos en JSON, logging y funciones auxiliares.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def logging(mensaje: str, archivo_log: str = "proyecto.log"):
    """
    Registra un mensaje en archivo de log.
    
    Args:
        mensaje (str): Mensaje a registrar
        archivo_log (str): Ruta del archivo de log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mensaje_formateado = f"[{timestamp}] {mensaje}"
    print(mensaje_formateado)
    
    try:
        with open(archivo_log, 'a', encoding='utf-8') as f:
            f.write(mensaje_formateado + "\n")
    except Exception as e:
        print(f"Error al escribir en log: {e}")


def guardar_pesos(parametros: Dict[str, np.ndarray], ruta: str):
    """
    Guarda los pesos de la red neuronal en formato JSON.
    
    Args:
        parametros (Dict): Diccionario con pesos y bias
        ruta (str): Ruta donde guardar
    """
    # Convertir arrays numpy a listas para JSON
    parametros_json = {}
    for clave, valor in parametros.items():
        if isinstance(valor, np.ndarray):
            parametros_json[clave] = valor.tolist()
        else:
            parametros_json[clave] = valor
    
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    
    with open(ruta, 'w') as f:
        json.dump(parametros_json, f, indent=2)
    
    logging(f"Pesos guardados en {ruta}")


def cargar_pesos(ruta: str) -> Dict[str, np.ndarray]:
    """
    Carga los pesos de la red neuronal desde JSON.
    
    Args:
        ruta (str): Ruta del archivo
        
    Returns:
        Dict: Diccionario con pesos y bias como arrays numpy
    """
    with open(ruta, 'r') as f:
        parametros_json = json.load(f)
    
    parametros = {}
    for clave, valor in parametros_json.items():
        parametros[clave] = np.array(valor)
    
    logging(f"Pesos cargados desde {ruta}")
    return parametros


def guardar_historial_csv(historial: Dict[str, list], ruta: str):
    """
    Guarda el historial de entrenamiento en CSV.
    
    Args:
        historial (Dict): Diccionario con listas de métricas
        ruta (str): Ruta donde guardar
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    
    with open(ruta, 'w') as f:
        # Escribir encabezados
        keys = list(historial.keys())
        f.write(','.join(keys) + '\n')
        
        # Escribir datos por fila
        n_rows = len(historial[keys[0]])
        for i in range(n_rows):
            valores = [str(historial[key][i]) for key in keys]
            f.write(','.join(valores) + '\n')
    
    logging(f"Historial guardado en {ruta}")


def cargar_historial_csv(ruta: str) -> Dict[str, list]:
    """
    Carga el historial de entrenamiento desde CSV.
    
    Args:
        ruta (str): Ruta del archivo
        
    Returns:
        Dict: Diccionario con listas de métricas
    """
    historial = {}
    
    with open(ruta, 'r') as f:
        lineas = f.readlines()
        
        # Primera línea: encabezados
        if len(lineas) > 0:
            headers = lineas[0].strip().split(',')
            for header in headers:
                historial[header] = []
            
            # Resto de líneas: datos
            for linea in lineas[1:]:
                valores = linea.strip().split(',')
                for header, valor in zip(headers, valores):
                    try:
                        historial[header].append(float(valor))
                    except ValueError:
                        historial[header].append(valor)
    
    logging(f"Historial cargado desde {ruta}")
    return historial


def crear_reporte_metricas(metricas: Dict[str, Any], ruta: str):
    """
    Crea un reporte de texto con las métricas.
    
    Args:
        metricas (Dict): Diccionario con métricas
        ruta (str): Ruta donde guardar el reporte
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    
    with open(ruta, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("REPORTE DE MÉTRICAS DEL MODELO\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for clave, valor in metricas.items():
            if clave == 'matriz_confusion':
                f.write(f"\n{clave}:\n")
                f.write(str(valor) + "\n")
            else:
                f.write(f"{clave}: {valor}\n")
    
    logging(f"Reporte de métricas guardado en {ruta}")
