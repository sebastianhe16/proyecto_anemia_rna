"""
Interfaz web para predicción de anemia.
Recibe una imagen del ojo (conjuntiva palpebral), extrae características,
usa el modelo entrenado y devuelve la predicción de hemoglobina + diagnóstico.
"""

import sys
import os
from pathlib import Path

# Agregar src al path
proyecto_root = Path(__file__).parent.parent
sys.path.insert(0, str(proyecto_root / "src"))
sys.path.insert(0, str(proyecto_root))

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from extraccion_caracteristicas import ExtractorCaracteristicas
from preprocesamiento import segmentar_roi_conjuntiva
from main import cargar_modelo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")

EXTENSIONES_PERMITIDAS = {"png", "jpg", "jpeg", "bmp"}

# Cargar modelo al iniciar la app
RUTA_MODELO = os.path.join(proyecto_root, "modelos", "modelo_final.npz")
modelo = None
escalador = None


def inicializar_modelo():
    """Carga el modelo entrenado si existe."""
    global modelo, escalador
    if os.path.exists(RUTA_MODELO):
        modelo, escalador = cargar_modelo(RUTA_MODELO)
        print(f"[OK] Modelo cargado desde: {RUTA_MODELO}")
    else:
        print(f"[AVISO] No se encontró el modelo en: {RUTA_MODELO}")
        print("        Ejecuta primero 'python main.py' para entrenar.")


def extension_valida(nombre_archivo):
    """Verifica que el archivo tenga una extensión de imagen válida."""
    return "." in nombre_archivo and \
           nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_PERMITIDAS


def procesar_imagen(ruta_imagen):
    """
    Pipeline completo: imagen -> segmentación -> extracción -> predicción.

    Intenta segmentar la conjuntiva. Si falla (porque la imagen ya viene
    recortada o no se detecta contorno), usa una máscara completa.
    """
    # Usar np.fromfile + imdecode para soportar rutas con caracteres
    # especiales (tildes, °, etc.) en Windows
    datos = np.fromfile(ruta_imagen, dtype=np.uint8)
    imagen_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen_bgr is None:
        raise ValueError("No se pudo leer la imagen.")

    # Intentar segmentación de la conjuntiva
    try:
        mascara, _ = segmentar_roi_conjuntiva(imagen_bgr, area_minima=500)
    except ValueError:
        # Si falla la segmentación, asumir que la imagen ya es la ROI
        mascara = np.ones(imagen_bgr.shape[:2], dtype=np.uint8) * 255

    # Extraer 12 características
    extractor = ExtractorCaracteristicas(
        umbral_inferior=20,
        umbral_superior=240,
        umbral_hue_normalizado=0.95,
    )
    vector = extractor.extraer(imagen_bgr, mascara)

    # Normalizar con el estandarizador del modelo
    X = vector.reshape(1, -1)
    X_std = escalador.transformar(X)

    # Predecir hemoglobina
    hb_predicha = float(modelo.predecir(X_std).flatten()[0])

    # Clasificación clínica
    es_anemia = hb_predicha < 11.0

    return {
        "hemoglobina_estimada": round(hb_predicha, 2),
        "diagnostico": "Anemia" if es_anemia else "No Anemia",
        "umbral_clinico": 11.0,
        "caracteristicas": {
            nombre: round(float(val), 4)
            for nombre, val in zip(ExtractorCaracteristicas.NOMBRES_CARACTERISTICAS, vector)
        },
    }


@app.route("/")
def index():
    """Página principal."""
    modelo_disponible = modelo is not None
    return render_template("index.html", modelo_disponible=modelo_disponible)


@app.route("/predecir", methods=["POST"])
def predecir():
    """Endpoint para recibir imagen y devolver predicción."""
    if modelo is None:
        return jsonify({"error": "Modelo no cargado. Ejecuta 'python main.py' primero."}), 503

    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400

    archivo = request.files["imagen"]
    if archivo.filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    if not extension_valida(archivo.filename):
        return jsonify({"error": "Formato no soportado. Usa PNG, JPG o BMP."}), 400

    # Guardar temporalmente
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    nombre_seguro = secure_filename(archivo.filename)
    ruta_temp = os.path.join(app.config["UPLOAD_FOLDER"], nombre_seguro)
    archivo.save(ruta_temp)

    try:
        resultado = procesar_imagen(ruta_temp)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": f"Error al procesar: {str(e)}"}), 500
    finally:
        # Limpiar archivo temporal
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)


if __name__ == "__main__":
    inicializar_modelo()
    print("\n" + "=" * 50)
    print(" Servidor web iniciado")
    print(" Abre http://127.0.0.1:5000 en tu navegador")
    print("=" * 50 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
