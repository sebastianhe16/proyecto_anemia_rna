# Documentación de Datos

## Fuente de Imágenes

Este documento describe el origen y estructura de las imágenes utilizadas en el proyecto de detección de anemia.

### Ubicación de Imágenes

Las imágenes se organizan en dos categorías:

- **anemico/**: Imágenes de pacientes con anemia diagnosticada
- **no_anemico/**: Imágenes de pacientes sin anemia

### Dataset CSV

El archivo `dataset.csv` contiene:

- **Características de imagen**: valores RGB, HSV y estadísticas
- **Características clínicas**: edad, sexo, nivel de hemoglobina
- **Etiqueta**: 0 (no anemia) o 1 (anemia)

### Preparación de Datos

1. Las imágenes se cargan desde las carpetas correspondientes
2. Se extraen características numéricas usando `extraccion_caracteristicas.py`
3. Las características se normalizan y se dividen en train/val/test
4. Los datos se guardan en `dataset.csv` para entrenamientos posteriores

### Metadatos Requeridos

Para cada imagen, es necesario contar con:

- Imagen médica (formato PNG, JPG, etc.)
- Edad del paciente (en años)
- Sexo del paciente (M/F)
- Nivel de hemoglobina (g/dL)
- Clasificación (anemia/no anemia)

---

**Nota**: Reemplaza esta información con los detalles específicos de tu dataset.
