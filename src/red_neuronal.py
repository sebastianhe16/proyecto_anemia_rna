"""
red_neuronal.py
-------------------------------------------------------------------------------
PROPOSITO:
    Este es el NUCLEO del trabajo: la implementacion DESDE CERO, sin
    ningun framework de Machine Learning ni de Deep Learning (nada de
    TensorFlow, PyTorch, Keras o Scikit-Learn), de un Perceptron
    Multicapa (red neuronal artificial feed-forward) para clasificacion
    binaria (0 = No anemico, 1 = Anemico).

    Este archivo usa UNICAMENTE el modulo "math" de la libreria
    estandar de Python (para exp, log, sqrt). No se usa numpy ni
    ninguna otra libreria externa en ningun punto de este archivo: los
    vectores y matrices se representan con listas comunes de Python, y
    todas las operaciones (multiplicaciones, sumas, derivadas) se
    programan explicitamente con bucles "for".

ARQUITECTURA (definida en el Documento 3 de referencia y replicada
exactamente segun lo acordado para este trabajo):

    Entrada (6 neuronas)
        |  r_promedio, g_promedio, b_promedio, s_promedio, edad, sexo
        v
    Capa oculta 1 (100 neuronas, activacion ReLU)
        v
    Capa oculta 2 (50 neuronas, activacion ReLU)
        v
    Capa de salida (1 neurona, activacion Sigmoide)
        v
    Salida: probabilidad de que el paciente sea anemico (0.0 a 1.0)

CONCEPTOS MATEMATICOS IMPLEMENTADOS A MANO:
    1. Forward propagation:  z = W*x + b ; activacion(z)
    2. Funcion de perdida:   Entropia cruzada binaria
    3. Backpropagation:      Regla de la cadena, capa por capa
    4. Optimizador Adam:     Promedios moviles de gradiente (m) y de
                             gradiente al cuadrado (v), con correccion
                             de sesgo, tal como se describe en el
                             Documento 3 (configuracion de referencia).
-------------------------------------------------------------------------------
"""

import math
import random


# ------------------------------------------------------------------------- #
# 1. FUNCIONES DE ACTIVACION Y SUS DERIVADAS (matematica pura con math)
# ------------------------------------------------------------------------- #
def relu(z):
    """
    Funcion de activacion ReLU (Rectified Linear Unit):
        ReLU(z) = max(0, z)

    Se usa en las dos capas ocultas (100 y 50 neuronas). Le da a la red
    la capacidad de aprender relaciones NO lineales entre las
    caracteristicas de entrada y la salida, algo que una simple
    combinacion lineal de pesos no podria lograr.
    """
    return z if z > 0.0 else 0.0


def derivada_relu(z):
    """
    Derivada de ReLU respecto a z:
        ReLU'(z) = 1 si z > 0
        ReLU'(z) = 0 si z <= 0

    Esta derivada es la que permite, durante el backpropagation, saber
    si el error debe "pasar" hacia atras a traves de esta neurona (si
    estaba activa, z>0) o si debe bloquearse por completo (si estaba
    apagada, z<=0).
    """
    return 1.0 if z > 0.0 else 0.0


def sigmoide(z):
    """
    Funcion de activacion Sigmoide:
        sigmoide(z) = 1 / (1 + e^(-z))

    Se usa UNICAMENTE en la neurona de salida, porque "aplasta"
    cualquier numero real al rango (0, 1), lo cual es ideal para
    interpretar el resultado como una probabilidad: que tan probable es,
    segun la red, que el paciente sea anemico.

    Se protege contra "overflow" matematico: si z es un numero muy
    negativo, e^(-z) podria crecer demasiado para que Python lo maneje
    como float. En ese caso, la sigmoide tiende a 0 de todas formas, asi
    que se devuelve 0.0 directamente sin intentar el calculo exacto.
    """
    if z < -700:  # e^700 ya es un numero astronomicamente grande
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


# ------------------------------------------------------------------------- #
# 2. FUNCION DE PERDIDA: ENTROPIA CRUZADA BINARIA
# ------------------------------------------------------------------------- #
def entropia_cruzada_binaria(etiqueta_real, prediccion):
    """
    Calcula la perdida (que tan "mal" predijo la red) para UN solo
    ejemplo, usando la formula de entropia cruzada binaria:

        L = -[ y*log(y_pred) + (1-y)*log(1-y_pred) ]

    Donde "y" es la etiqueta real (0 o 1) y "y_pred" es la probabilidad
    que la red asigno a la clase 1 (anemico).

    Se recorta (clip) la prediccion a un rango seguro [epsilon, 1-epsilon]
    para evitar calcular log(0), que matematicamente no esta definido y
    haria que Python lance un error.
    """
    epsilon = 1e-12
    prediccion_segura = min(max(prediccion, epsilon), 1.0 - epsilon)

    if etiqueta_real == 1:
        return -math.log(prediccion_segura)
    else:
        return -math.log(1.0 - prediccion_segura)


# ------------------------------------------------------------------------- #
# 3. CLASE CapaDensa: UNA CAPA DE LA RED (PESOS + BIAS + ACTIVACION)
# ------------------------------------------------------------------------- #
class CapaDensa:
    """
    Representa una capa "totalmente conectada" (dense / fully connected):
    cada neurona de esta capa recibe como entrada TODAS las salidas de
    la capa anterior.

    Atributos principales:
        pesos[j][i]  -> peso que conecta la entrada i con la neurona j
        bias[j]      -> bias (sesgo) de la neurona j
        activacion   -> "relu" o "sigmoide", segun el rol de esta capa

    Los pesos se representan como una lista de listas (matriz), y el
    bias como una lista simple. No se usa numpy: todas las
    multiplicaciones y sumas se hacen con bucles "for" explicitos.
    """

    def __init__(self, cantidad_entradas, cantidad_neuronas, tipo_activacion):
        self.cantidad_entradas = cantidad_entradas
        self.cantidad_neuronas = cantidad_neuronas
        self.tipo_activacion = tipo_activacion  # "relu" o "sigmoide"

        # Inicializacion de pesos: se usan valores aleatorios pequeños
        # (inicializacion tipo "He", recomendada para capas con ReLU).
        # Esto evita que todas las neuronas arranquen con el mismo
        # valor (lo cual les impediria aprender cosas distintas entre
        # si) y evita que los valores iniciales sean demasiado grandes
        # (lo cual podria hacer que la red "explote" numericamente).
        limite = math.sqrt(2.0 / cantidad_entradas)
        self.pesos = [
            [random.uniform(-limite, limite) for _ in range(cantidad_entradas)]
            for _ in range(cantidad_neuronas)
        ]
        self.bias = [0.0 for _ in range(cantidad_neuronas)]

        # --- Variables auxiliares para Adam (una por cada peso y bias) ---
        # m: promedio movil del gradiente (primer momento)
        # v: promedio movil del gradiente al cuadrado (segundo momento)
        self.m_pesos = [[0.0] * cantidad_entradas for _ in range(cantidad_neuronas)]
        self.v_pesos = [[0.0] * cantidad_entradas for _ in range(cantidad_neuronas)]
        self.m_bias = [0.0 for _ in range(cantidad_neuronas)]
        self.v_bias = [0.0 for _ in range(cantidad_neuronas)]

        # --- Variables guardadas durante el forward, para usarlas en backward ---
        self.ultima_entrada = None    # x: lo que entro a esta capa
        self.ultimas_z = None         # z: suma ponderada antes de la activacion
        self.ultima_salida = None     # a: salida despues de la activacion

    # --------------------------------------------------------------- #
    def forward(self, entrada):
        """
        Calcula la salida de esta capa dado un vector de entrada.

        Para cada neurona j:
            z_j = sum_i( pesos[j][i] * entrada[i] ) + bias[j]
            a_j = activacion(z_j)

        Guarda "entrada", "z" y "a" porque backward() los necesitara
        despues para calcular las derivadas correctamente.
        """
        self.ultima_entrada = entrada
        self.ultimas_z = []
        self.ultima_salida = []

        for j in range(self.cantidad_neuronas):
            suma_ponderada = self.bias[j]
            for i in range(self.cantidad_entradas):
                suma_ponderada += self.pesos[j][i] * entrada[i]

            self.ultimas_z.append(suma_ponderada)

            if self.tipo_activacion == "relu":
                self.ultima_salida.append(relu(suma_ponderada))
            elif self.tipo_activacion == "sigmoide":
                self.ultima_salida.append(sigmoide(suma_ponderada))
            else:
                raise ValueError(f"Tipo de activacion no soportado: {self.tipo_activacion}")

        return self.ultima_salida

    # --------------------------------------------------------------- #
    def backward(self, gradiente_salida):
        """
        Dado el "delta" que llega desde la capa siguiente (cuanto
        contribuyo el error de esta capa al error final), calcula:
            1. El gradiente respecto a cada peso y cada bias de ESTA capa
               (para poder actualizarlos despues con Adam).
            2. El "delta" que hay que propagar hacia la capa ANTERIOR
               (para que esa capa pueda hacer lo mismo).

        gradiente_salida[j] representa dL/dz_j para la neurona j de
        ESTA capa (ya incluye la derivada de la activacion si esta
        capa es la de salida; para capas ocultas, se calcula aqui
        mismo multiplicando por la derivada de ReLU).
        """
        # Si esta capa usa ReLU, hay que multiplicar el gradiente que
        # llega por la derivada de ReLU evaluada en el z guardado
        # durante el forward. (Si la capa es la de salida con sigmoide
        # + entropia cruzada, el gradiente_salida que recibe ya viene
        # simplificado como (prediccion - etiqueta_real), calculado en
        # RedNeuronal.backward(), por lo que aqui NO se vuelve a
        # multiplicar por la derivada de sigmoide).
        deltas = []
        for j in range(self.cantidad_neuronas):
            if self.tipo_activacion == "relu":
                deltas.append(gradiente_salida[j] * derivada_relu(self.ultimas_z[j]))
            else:  # "sigmoide" (capa de salida): el gradiente ya viene listo
                deltas.append(gradiente_salida[j])

        # Gradiente respecto a los pesos y al bias de esta capa.
        self.gradiente_pesos = [
            [deltas[j] * self.ultima_entrada[i] for i in range(self.cantidad_entradas)]
            for j in range(self.cantidad_neuronas)
        ]
        self.gradiente_bias = list(deltas)

        # Delta que se propaga hacia la capa anterior: para cada
        # entrada i, se suma la contribucion de todas las neuronas j
        # de esta capa que usaron esa entrada (regla de la cadena).
        gradiente_entrada = [0.0 for _ in range(self.cantidad_entradas)]
        for i in range(self.cantidad_entradas):
            suma = 0.0
            for j in range(self.cantidad_neuronas):
                suma += deltas[j] * self.pesos[j][i]
            gradiente_entrada[i] = suma

        return gradiente_entrada

    # --------------------------------------------------------------- #
    def actualizar_pesos_adam(self, tasa_aprendizaje, beta1, beta2, epsilon, paso_tiempo):
        """
        Aplica el optimizador Adam para actualizar todos los pesos y
        bias de esta capa, usando los gradientes calculados en el
        ultimo backward().

        Formulas (aplicadas individualmente a cada peso/bias):
            m_t = beta1*m_(t-1) + (1-beta1)*g_t
            v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
            m_corregido = m_t / (1 - beta1^t)
            v_corregido = v_t / (1 - beta2^t)
            w_t = w_(t-1) - tasa_aprendizaje * m_corregido / (sqrt(v_corregido) + epsilon)
        """
        for j in range(self.cantidad_neuronas):
            for i in range(self.cantidad_entradas):
                gradiente = self.gradiente_pesos[j][i]

                self.m_pesos[j][i] = beta1 * self.m_pesos[j][i] + (1 - beta1) * gradiente
                self.v_pesos[j][i] = beta2 * self.v_pesos[j][i] + (1 - beta2) * (gradiente ** 2)

                m_corregido = self.m_pesos[j][i] / (1 - beta1 ** paso_tiempo)
                v_corregido = self.v_pesos[j][i] / (1 - beta2 ** paso_tiempo)

                self.pesos[j][i] -= tasa_aprendizaje * m_corregido / (math.sqrt(v_corregido) + epsilon)

            # Misma logica de Adam, pero para el bias de la neurona j.
            gradiente_b = self.gradiente_bias[j]
            self.m_bias[j] = beta1 * self.m_bias[j] + (1 - beta1) * gradiente_b
            self.v_bias[j] = beta2 * self.v_bias[j] + (1 - beta2) * (gradiente_b ** 2)

            m_bias_corregido = self.m_bias[j] / (1 - beta1 ** paso_tiempo)
            v_bias_corregido = self.v_bias[j] / (1 - beta2 ** paso_tiempo)

            self.bias[j] -= tasa_aprendizaje * m_bias_corregido / (math.sqrt(v_bias_corregido) + epsilon)


# ------------------------------------------------------------------------- #
# 4. CLASE RedNeuronal: JUNTA LAS 3 CAPAS Y ORQUESTA TODO EL PROCESO
# ------------------------------------------------------------------------- #
class RedNeuronal:
    """
    Perceptron Multicapa completo para el problema de deteccion de
    anemia: 6 entradas -> 100 (ReLU) -> 50 (ReLU) -> 1 (Sigmoide).

    Hiperparametros de Adam (valores estandar de la literatura, los
    mismos que usa por defecto, por ejemplo, TensorFlow/Keras):
        tasa_aprendizaje = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
    """

    def __init__(self, arquitectura=None, cantidad_entradas=6, neuronas_capa1=100, neuronas_capa2=50,
                 tasa_aprendizaje=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if arquitectura is not None:
            cantidad_entradas = arquitectura[0]
            neuronas_capa1 = arquitectura[1]
            neuronas_capa2 = arquitectura[2]

        self.capa1 = CapaDensa(cantidad_entradas, neuronas_capa1, "relu")
        self.capa2 = CapaDensa(neuronas_capa1, neuronas_capa2, "relu")
        self.capa_salida = CapaDensa(neuronas_capa2, 1, "sigmoide")

        self.tasa_aprendizaje = tasa_aprendizaje
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Contador de pasos de actualizacion ya realizados. Es
        # necesario para la correccion de sesgo de Adam (beta1^t y
        # beta2^t dependen de "t", el numero de actualizacion actual).
        self.paso_tiempo = 0

    # --------------------------------------------------------------- #
    def forward(self, entrada):
        """Propaga uno o varios ejemplos a través de la red."""
        if hasattr(entrada, "shape") and len(entrada.shape) > 1:
            return [self._forward_un_ejemplo(list(fila)) for fila in entrada]

        if isinstance(entrada, (list, tuple)) and len(entrada) > 0 and isinstance(entrada[0], (list, tuple)):
            return [self._forward_un_ejemplo(list(fila)) for fila in entrada]

        return self._forward_un_ejemplo(list(entrada))

    def _forward_un_ejemplo(self, entrada):
        salida_capa1 = self.capa1.forward(entrada)
        salida_capa2 = self.capa2.forward(salida_capa1)
        salida_final = self.capa_salida.forward(salida_capa2)
        return salida_final[0]

    def backward(self, gradiente_salida):
        gradiente_capa2 = self.capa_salida.backward(gradiente_salida)
        gradiente_capa1 = self.capa2.backward(gradiente_capa2)
        self.capa1.backward(gradiente_capa1)

    def actualizar_pesos_adam(self):
        self.paso_tiempo += 1
        self.capa1.actualizar_pesos_adam(self.tasa_aprendizaje, self.beta1, self.beta2,
                                          self.epsilon, self.paso_tiempo)
        self.capa2.actualizar_pesos_adam(self.tasa_aprendizaje, self.beta1, self.beta2,
                                          self.epsilon, self.paso_tiempo)
        self.capa_salida.actualizar_pesos_adam(self.tasa_aprendizaje, self.beta1, self.beta2,
                                                self.epsilon, self.paso_tiempo)

    # --------------------------------------------------------------- #
    def entrenar_un_ejemplo(self, entrada, etiqueta_real):
        """
        Realiza UN paso completo de entrenamiento para un solo ejemplo:
            1. Forward propagation (calcula la prediccion)
            2. Calcula la perdida (que tan mal predijo)
            3. Backpropagation (calcula los gradientes de las 3 capas)
            4. Actualiza los pesos de las 3 capas usando Adam

        Devuelve la perdida de este ejemplo, para poder promediarla
        despues y graficar la curva de entrenamiento.
        """
        prediccion = self.forward(entrada)
        perdida = entropia_cruzada_binaria(etiqueta_real, prediccion)

        # Derivada de la perdida respecto a "z" de la neurona de salida.
        # Gracias a que se usa sigmoide + entropia cruzada juntas, esta
        # derivada se simplifica matematicamente a una resta muy simple:
        #     dL/dz_salida = prediccion - etiqueta_real
        gradiente_salida = [prediccion - etiqueta_real]

        # Backpropagation: el error se propaga de la capa de salida
        # hacia la capa 2, y de la capa 2 hacia la capa 1.
        gradiente_capa2 = self.capa_salida.backward(gradiente_salida)
        gradiente_capa1 = self.capa2.backward(gradiente_capa2)
        self.capa1.backward(gradiente_capa1)

        self.actualizar_pesos_adam()

        return perdida

    # --------------------------------------------------------------- #
    def obtener_parametros(self):
        """Devuelve los pesos y sesgos de las capas en formato serializable."""
        return {
            "capa1_pesos": self.capa1.pesos,
            "capa1_bias": self.capa1.bias,
            "capa2_pesos": self.capa2.pesos,
            "capa2_bias": self.capa2.bias,
            "capa_salida_pesos": self.capa_salida.pesos,
            "capa_salida_bias": self.capa_salida.bias,
        }

    def establecer_parametros(self, parametros):
        """Recupera pesos y sesgos previamente guardados."""
        self.capa1.pesos = parametros["capa1_pesos"]
        self.capa1.bias = parametros["capa1_bias"]
        self.capa2.pesos = parametros["capa2_pesos"]
        self.capa2.bias = parametros["capa2_bias"]
        self.capa_salida.pesos = parametros["capa_salida_pesos"]
        self.capa_salida.bias = parametros["capa_salida_bias"]

    def predecir(self, entrada, umbral=0.5):
        """
        Usa la red YA ENTRENADA para clasificar un ejemplo nuevo, sin
        modificar ningun peso. Devuelve una tupla:
            (probabilidad, clase_predicha)

        clase_predicha es 1 (anemico) si la probabilidad supera el
        umbral (por defecto 0.5), o 0 (no anemico) en caso contrario.
        """
        probabilidad = self.forward(entrada)
        clase_predicha = 1 if probabilidad >= umbral else 0
        return probabilidad, clase_predicha


# ------------------------------------------------------------------------- #
# 5. PRUEBA RAPIDA AL EJECUTAR ESTE ARCHIVO DIRECTAMENTE
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Pequeña prueba de humo: verifica que la red se puede crear, hacer
    # forward, entrenar varios pasos y que la perdida tienda a bajar.
    random.seed(42)
    red = RedNeuronal()

    entrada_prueba = [0.8, 0.2, 0.3, 0.7, 0.5, 1.0]
    etiqueta_prueba = 1

    print("Prediccion ANTES de entrenar:", red.forward(entrada_prueba))

    for epoca in range(200):
        perdida = red.entrenar_un_ejemplo(entrada_prueba, etiqueta_prueba)
        if epoca % 50 == 0:
            print(f"Epoca {epoca}: perdida = {perdida:.6f}")

    print("Prediccion DESPUES de entrenar:", red.forward(entrada_prueba))
    print("(deberia acercarse bastante a 1.0, que es la etiqueta real)")
