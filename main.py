"""
Punto de entrada del proyecto.
Menú principal para ejecutar el pipeline completo o pasos individuales.
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utilidades import logging


def menu_principal():
    """Menú principal del programa."""
    
    print("\n" + "="*60)
    print("SISTEMA DE DETECCIÓN DE ANEMIA MEDIANTE RNA")
    print("="*60)
    print("\nOpciones disponibles:\n")
    print("1. Preparar datos (extraer características y crear dataset.csv)")
    print("2. Entrenar modelo")
    print("3. Evaluar modelo")
    print("4. Pipeline completo (preparar -> entrenar -> evaluar)")
    print("5. Ver historial de entrenamiento")
    print("6. Salir")
    print("\n" + "-"*60)
    
    opcion = input("\nSelecciona una opción (1-6): ").strip()
    
    return opcion


def preparar_datos():
    """Prepara los datos extrayendo características de imágenes."""
    logging("="*60)
    logging("INICIANDO PREPARACIÓN DE DATOS")
    logging("="*60)
    
    print("\nEsta función cargará imágenes de:")
    print("  - data/raw_images/anemico/")
    print("  - data/raw_images/no_anemico/")
    print("\nY generará: data/dataset.csv")
    print("\n[Implementar lógica de carga y extracción de características]")
    
    logging("Preparación de datos completada")


def entrenar_modelo():
    """Entrena el modelo neural."""
    logging("="*60)
    logging("INICIANDO ENTRENAMIENTO")
    logging("="*60)
    
    print("\nEntrenando modelo...")
    print("[Implementar lógica de entrenamiento]")
    
    logging("Entrenamiento completado")


def evaluar_modelo():
    """Evalúa el modelo entrenado."""
    logging("="*60)
    logging("INICIANDO EVALUACIÓN")
    logging("="*60)
    
    print("\nEvaluando modelo...")
    print("[Implementar lógica de evaluación]")
    
    logging("Evaluación completada")


def pipeline_completo():
    """Ejecuta el pipeline completo."""
    logging("="*60)
    logging("INICIANDO PIPELINE COMPLETO")
    logging("="*60)
    
    print("\nEjecutando pipeline completo...\n")
    
    print("Paso 1: Preparar datos")
    preparar_datos()
    
    print("\n\nPaso 2: Entrenar modelo")
    entrenar_modelo()
    
    print("\n\nPaso 3: Evaluar modelo")
    evaluar_modelo()
    
    logging("Pipeline completo finalizado")


def ver_historial():
    """Muestra el historial de entrenamiento."""
    print("\nMostrando historial de entrenamiento...")
    
    ruta_historial = Path("resultados/historial_entrenamiento.csv")
    
    if ruta_historial.exists():
        with open(ruta_historial, 'r') as f:
            contenido = f.read()
            print(contenido)
    else:
        print("No se encontró historial. Ejecuta primero el entrenamiento.")


def main():
    """Función principal."""
    logging("Aplicación iniciada")
    
    while True:
        opcion = menu_principal()
        
        if opcion == '1':
            preparar_datos()
        elif opcion == '2':
            entrenar_modelo()
        elif opcion == '3':
            evaluar_modelo()
        elif opcion == '4':
            pipeline_completo()
        elif opcion == '5':
            ver_historial()
        elif opcion == '6':
            print("\n¡Hasta luego!")
            logging("Aplicación finalizada")
            break
        else:
            print("\nOpción no válida. Por favor selecciona 1-6.")
        
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()
