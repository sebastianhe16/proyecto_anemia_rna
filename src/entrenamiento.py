"""
Orquestación del entrenamiento.
Carga dataset.csv, entrena la red y guarda los pesos.
"""

import numpy as np
import json
from typing import Tuple
from pathlib import Path

from red_neuronal import RedNeuronal
from metricas import evaluar_modelo, perdida_entropia_cruzada
from utilidades import guardar_pesos, cargar_pesos, logging


class Entrenador:
    """Clase para entrenar la red neuronal."""
    
    def __init__(self, arquitectura: list, lr=0.001, epochs=100, batch_size=32):
        """
        Inicializa el entrenador.
        
        Args:
            arquitectura (list): Arquitectura de la red (ej: [20, 32, 16, 1])
            lr (float): Learning rate
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
        """
        self.red = RedNeuronal(arquitectura, lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.historial = {
            'perdida_entrenamiento': [],
            'perdida_validacion': [],
            'accuracy_entrenamiento': [],
            'accuracy_validacion': []
        }
    
    def entrenar_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Entrena una época.
        
        Args:
            X_train (np.ndarray): Datos de entrenamiento
            y_train (np.ndarray): Etiquetas de entrenamiento
            
        Returns:
            float: Pérdida promedio de la época
        """
        n_muestras = len(X_train)
        n_batches = (n_muestras + self.batch_size - 1) // self.batch_size
        
        indices = np.random.permutation(n_muestras)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        perdida_total = 0
        
        for i in range(n_batches):
            inicio = i * self.batch_size
            fin = min(inicio + self.batch_size, n_muestras)
            
            X_batch = X_shuffled[inicio:fin]
            y_batch = y_shuffled[inicio:fin]
            
            # Forward pass
            y_pred = self.red.forward(X_batch)
            
            # Calcular pérdida
            perdida = perdida_entropia_cruzada(y_batch, y_pred.flatten())
            perdida_total += perdida
            
            # Backward pass
            dA = (y_pred.flatten() - y_batch) / len(y_batch)
            self.red.backward(dA.reshape(-1, 1))
            
            # Actualizar pesos con Adam
            self.red.actualizar_pesos_adam()
        
        return perdida_total / n_batches
    
    def evaluar(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """
        Evalúa el modelo en datos de validación.
        
        Args:
            X_val (np.ndarray): Datos de validación
            y_val (np.ndarray): Etiquetas de validación
            
        Returns:
            Tuple: (pérdida, accuracy)
        """
        y_pred_proba = self.red.forward(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        perdida = perdida_entropia_cruzada(y_val, y_pred_proba)
        metricas = evaluar_modelo(y_val, y_pred, y_pred_proba)
        accuracy = metricas['accuracy']
        
        return perdida, accuracy
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, verbose=True) -> dict:
        """
        Entrena el modelo durante varias épocas.
        
        Args:
            X_train (np.ndarray): Datos de entrenamiento
            y_train (np.ndarray): Etiquetas de entrenamiento
            X_val (np.ndarray): Datos de validación
            y_val (np.ndarray): Etiquetas de validación
            verbose (bool): Mostrar progreso
            
        Returns:
            dict: Historial de entrenamiento
        """
        for epoch in range(self.epochs):
            # Entrenar
            perdida_train = self.entrenar_epoch(X_train, y_train)
            
            # Evaluar
            perdida_val, accuracy_val = self.evaluar(X_val, y_val)
            
            # Guardar historial
            self.historial['perdida_entrenamiento'].append(perdida_train)
            self.historial['perdida_validacion'].append(perdida_val)
            
            # Accuracy en entrenamiento
            y_pred_train = (self.red.forward(X_train).flatten() > 0.5).astype(int)
            accuracy_train = np.mean(y_train == y_pred_train)
            self.historial['accuracy_entrenamiento'].append(accuracy_train)
            self.historial['accuracy_validacion'].append(accuracy_val)
            
            if verbose and (epoch + 1) % 10 == 0:
                logging(f"Época {epoch+1}/{self.epochs} - "
                       f"Loss: {perdida_train:.4f}, Val Loss: {perdida_val:.4f}, "
                       f"Acc: {accuracy_train:.4f}, Val Acc: {accuracy_val:.4f}")
        
        return self.historial
    
    def guardar_modelo(self, ruta: str):
        """
        Guarda los pesos entrenados.
        
        Args:
            ruta (str): Ruta donde guardar los pesos
        """
        parametros = self.red.obtener_parametros()
        guardar_pesos(parametros, ruta)
        logging(f"Modelo guardado en {ruta}")
    
    def cargar_modelo(self, ruta: str):
        """
        Carga los pesos entrenados.
        
        Args:
            ruta (str): Ruta de donde cargar los pesos
        """
        parametros = cargar_pesos(ruta)
        self.red.establecer_parametros(parametros)
        logging(f"Modelo cargado desde {ruta}")
