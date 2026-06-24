"""
Red Neuronal implementada desde cero.
Incluye forward pass, backpropagation y optimizador Adam - solo matemática.
"""

import numpy as np
from typing import List, Tuple


class CapaLineal:
    """Capa lineal (fully connected) de la red neuronal."""
    
    def __init__(self, entrada_dim: int, salida_dim: int, seed=42):
        """
        Inicializa los pesos y bias.
        
        Args:
            entrada_dim (int): Dimensión de entrada
            salida_dim (int): Dimensión de salida
            seed (int): Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.pesos = np.random.randn(entrada_dim, salida_dim) * 0.01
        self.bias = np.zeros((1, salida_dim))
        self.entrada = None
        self.salida = None
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: Z = X @ W + b
        
        Args:
            X (np.ndarray): Entrada (batch_size, entrada_dim)
            
        Returns:
            np.ndarray: Salida (batch_size, salida_dim)
        """
        self.entrada = X
        self.salida = X @ self.pesos + self.bias
        return self.salida
    
    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        Backward pass: calcula gradientes.
        
        Args:
            dz (np.ndarray): Gradiente de pérdida respecto a salida
            
        Returns:
            np.ndarray: Gradiente respecto a entrada
        """
        m = self.entrada.shape[0]
        
        self.dW = self.entrada.T @ dz / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        dx = dz @ self.pesos.T
        
        return dx


class Activacion:
    """Función de activación ReLU."""
    
    def __init__(self):
        self.entrada = None
        
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """ReLU: max(0, Z)"""
        self.entrada = Z
        return np.maximum(0, Z)
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Derivada de ReLU"""
        return dA * (self.entrada > 0)


class Sigmoid:
    """Función de activación Sigmoid para clasificación binaria."""
    
    def __init__(self):
        self.salida = None
        
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Sigmoid: 1 / (1 + e^-Z)"""
        self.salida = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        return self.salida
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Derivada de Sigmoid: A * (1 - A)"""
        return dA * self.salida * (1 - self.salida)


class RedNeuronal:
    """Red Neuronal completa con forward, backward y optimización Adam."""
    
    def __init__(self, capas: List[int], lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Inicializa la red neuronal.
        
        Args:
            capas (List[int]): Lista con dimensiones de cada capa
            lr (float): Learning rate
            beta1 (float): Parámetro beta1 de Adam
            beta2 (float): Parámetro beta2 de Adam
            epsilon (float): Parámetro epsilon de Adam
        """
        self.capas_lineales = []
        self.activaciones = []
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Contador para Adam
        
        # Construir capas
        for i in range(len(capas) - 1):
            self.capas_lineales.append(CapaLineal(capas[i], capas[i+1]))
            
            # Última capa tiene Sigmoid, otras tienen ReLU
            if i == len(capas) - 2:
                self.activaciones.append(Sigmoid())
            else:
                self.activaciones.append(Activacion())
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass a través de toda la red."""
        A = X
        for capa, activacion in zip(self.capas_lineales, self.activaciones):
            Z = capa.forward(A)
            A = activacion.forward(Z)
        return A
    
    def backward(self, dA_final: np.ndarray):
        """Backward pass a través de toda la red."""
        dA = dA_final
        for i in reversed(range(len(self.capas_lineales))):
            dZ = self.activaciones[i].backward(dA)
            dA = self.capas_lineales[i].backward(dZ)
    
    def actualizar_pesos_adam(self):
        """Actualiza pesos usando optimizador Adam."""
        self.t += 1
        
        for capa in self.capas_lineales:
            if not hasattr(capa, 'm_W'):
                capa.m_W = np.zeros_like(capa.pesos)
                capa.v_W = np.zeros_like(capa.pesos)
                capa.m_b = np.zeros_like(capa.bias)
                capa.v_b = np.zeros_like(capa.bias)
            
            # Actualizar momentos para W
            capa.m_W = self.beta1 * capa.m_W + (1 - self.beta1) * capa.dW
            capa.v_W = self.beta2 * capa.v_W + (1 - self.beta2) * (capa.dW ** 2)
            
            # Actualizar momentos para b
            capa.m_b = self.beta1 * capa.m_b + (1 - self.beta1) * capa.db
            capa.v_b = self.beta2 * capa.v_b + (1 - self.beta2) * (capa.db ** 2)
            
            # Correcciones de sesgo
            m_W_corregido = capa.m_W / (1 - self.beta1 ** self.t)
            v_W_corregido = capa.v_W / (1 - self.beta2 ** self.t)
            m_b_corregido = capa.m_b / (1 - self.beta1 ** self.t)
            v_b_corregido = capa.v_b / (1 - self.beta2 ** self.t)
            
            # Actualizar pesos
            capa.pesos -= self.lr * m_W_corregido / (np.sqrt(v_W_corregido) + self.epsilon)
            capa.bias -= self.lr * m_b_corregido / (np.sqrt(v_b_corregido) + self.epsilon)
    
    def obtener_parametros(self) -> dict:
        """Obtiene los pesos y bias de todas las capas."""
        parametros = {}
        for i, capa in enumerate(self.capas_lineales):
            parametros[f'W{i}'] = capa.pesos
            parametros[f'b{i}'] = capa.bias
        return parametros
    
    def establecer_parametros(self, parametros: dict):
        """Establece los pesos y bias de todas las capas."""
        for i, capa in enumerate(self.capas_lineales):
            capa.pesos = parametros[f'W{i}']
            capa.bias = parametros[f'b{i}']
