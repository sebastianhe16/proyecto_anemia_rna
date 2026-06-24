"""
Preprocesamiento de datos.
Incluye normalización Min-Max, split train/val/test y balanceo de datos.
"""

import numpy as np
from typing import Tuple, List


def normalizar_minmax(X, min_vals=None, max_vals=None):
    """
    Normaliza datos usando Min-Max scaling a rango [0, 1].
    
    Args:
        X (np.ndarray): Datos a normalizar (muestras, características)
        min_vals (np.ndarray, optional): Valores mínimos precalculados
        max_vals (np.ndarray, optional): Valores máximos precalculados
        
    Returns:
        np.ndarray: Datos normalizados
        np.ndarray: Valores mínimos usados
        np.ndarray: Valores máximos usados
    """
    if min_vals is None:
        min_vals = np.min(X, axis=0)
    if max_vals is None:
        max_vals = np.max(X, axis=0)
    
    # Evitar división por cero
    rango = max_vals - min_vals
    rango[rango == 0] = 1
    
    X_normalizado = (X - min_vals) / rango
    
    return X_normalizado, min_vals, max_vals


def dividir_train_val_test(X, y, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        X (np.ndarray): Características (muestras, características)
        y (np.ndarray): Etiquetas (muestras,)
        ratios (tuple): Proporciones (train, val, test)
        seed (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    
    train_size = int(n * ratios[0])
    val_size = int(n * ratios[1])
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])


def balancear_clases(X, y, metodo='submuestreo'):
    """
    Balancea las clases en el dataset.
    
    Args:
        X (np.ndarray): Características
        y (np.ndarray): Etiquetas
        metodo (str): 'submuestreo' o 'sobremuestreo'
        
    Returns:
        np.ndarray: Características balanceadas
        np.ndarray: Etiquetas balanceadas
    """
    clases_unicas, conteos = np.unique(y, return_counts=True)
    
    if metodo == 'submuestreo':
        # Reducir la clase mayoritaria
        min_muestras = np.min(conteos)
        indices_balanceados = []
        
        for clase in clases_unicas:
            indices_clase = np.where(y == clase)[0]
            indices_seleccionados = np.random.choice(indices_clase, min_muestras, replace=False)
            indices_balanceados.extend(indices_seleccionados)
            
    elif metodo == 'sobremuestreo':
        # Aumentar la clase minoritaria
        max_muestras = np.max(conteos)
        indices_balanceados = []
        
        for clase in clases_unicas:
            indices_clase = np.where(y == clase)[0]
            indices_seleccionados = np.random.choice(indices_clase, max_muestras, replace=True)
            indices_balanceados.extend(indices_seleccionados)
    
    indices_balanceados = np.array(indices_balanceados)
    np.random.shuffle(indices_balanceados)
    
    return X[indices_balanceados], y[indices_balanceados]
