import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utilidades import logging, guardar_historial_csv, crear_reporte_metricas
from preprocesamiento import construir_conjuntos_preprocesados, balancear_clases, dividir_train_val_test, normalizar_minmax
from extraccion_caracteristicas import construir_dataset
from entrenamiento import Entrenador
from metricas import evaluar_modelo as calcular_metricas


def menu_principal():
    print("\n" + "=" * 60)
    print("SISTEMA DE DETECCIÓN DE ANEMIA MEDIANTE RNA")
    print("=" * 60)
    print("\nOpciones disponibles:\n")
    print("1. Preparar datos")
    print("2. Entrenar modelo")
    print("3. Evaluar modelo")
    print("4. Pipeline completo")
    print("5. Ver historial")
    print("6. Salir")
    print("\n" + "-" * 60)
    opcion = input("\nSelecciona una opción (1-6): ").strip()
    return opcion


def preparar_datos():
    logging("INICIANDO PREPARACIÓN DE DATOS")
    if not Path("data/dataset.csv").exists():
        construir_dataset()
    construir_conjuntos_preprocesados()
    logging("Preparación de datos completada.")


def cargar_datos_preprocesados():
    train = np.loadtxt("data/dataset_train.csv", delimiter=",", skiprows=1)
    val = np.loadtxt("data/dataset_val.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("data/dataset_test.csv", delimiter=",", skiprows=1)
    return train[:, :-1], val[:, :-1], test[:, :-1], train[:, -1], val[:, -1], test[:, -1]


def entrenar_modelo():
    logging("INICIANDO ENTRENAMIENTO")
    if not Path("data/dataset_train.csv").exists():
        preparar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos_preprocesados()
    X_bal, y_bal = balancear_clases(X_train, y_train, metodo="submuestreo")
    X_norm, _, _ = normalizar_minmax(X_bal)
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = dividir_train_val_test(X_norm, y_bal)
    np.savetxt("data/X_test.csv", X_test2, delimiter=",")
    np.savetxt("data/y_test.csv", y_test2, delimiter=",")
    arquitectura = [X_train2.shape[1], 100, 50, 1]
    entrenador = Entrenador(arquitectura, lr=0.005, epochs=60, batch_size=16)
    historial = entrenador.entrenar(X_train2, y_train2, X_val2, y_val2, verbose=True)
    entrenador.guardar_modelo("modelos/pesos_entrenados.json")
    guardar_historial_csv(historial, "resultados/historial_entrenamiento.csv")
    logging("Entrenamiento completado.")


def evaluar_modelo():
    logging("INICIANDO EVALUACIÓN")
    if not Path("modelos/pesos_entrenados.json").exists() or not Path("data/X_test.csv").exists():
        print("Error: faltan los pesos entrenados o el conjunto de prueba.")
        return
    X_test = np.loadtxt("data/X_test.csv", delimiter=",")
    y_test = np.loadtxt("data/y_test.csv", delimiter=",")
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)
    arquitectura = [X_test.shape[1], 100, 50, 1]
    entrenador = Entrenador(arquitectura)
    entrenador.cargar_modelo("modelos/pesos_entrenados.json")
    y_pred_proba = np.array(entrenador.red.forward(X_test), dtype=float).reshape(-1)
    y_pred = (y_pred_proba > 0.5).astype(int)
    metricas = calcular_metrica_completa(y_test, y_pred, y_pred_proba)
    print("\n" + "-" * 30)
    print("MÉTRICAS OBTENIDAS EN PRUEBA:")
    print(f"Exactitud (Accuracy): {metricas['accuracy']:.4f}")
    print(f"Precisión (Precision): {metricas['precision']:.4f}")
    print(f"Sensibilidad (Recall): {metricas['recall']:.4f}")
    print(f"F1-Score: {metricas['f1']:.4f}")
    print("Matriz de Confusión:")
    print(metricas['matriz_confusion'])
    print("-" * 30)
    crear_reporte_metricas(metricas, "resultados/reporte_metricas.txt")
    logging("Evaluación completada.")


def calcular_metrica_completa(y_real, y_pred, y_pred_proba):
    return calcular_metricas(y_real, y_pred, y_pred_proba)


def pipeline_completo():
    preparar_datos()
    entrenar_modelo()
    evaluar_modelo()


def ver_historial():
    ruta_historial = Path("resultados/historial_entrenamiento.csv")
    if ruta_historial.exists():
        print(ruta_historial.read_text(encoding="utf-8"))
    else:
        print("No existe historial todavía.")


def main():
    logging("Aplicación iniciada")
    while True:
        opcion = menu_principal()
        if opcion == "1":
            preparar_datos()
        elif opcion == "2":
            entrenar_modelo()
        elif opcion == "3":
            evaluar_modelo()
        elif opcion == "4":
            pipeline_completo()
        elif opcion == "5":
            ver_historial()
        elif opcion == "6":
            print("\n¡Hasta luego!")
            logging("Aplicación finalizada")
            break
        else:
            print("\nOpción no válida.")
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()
