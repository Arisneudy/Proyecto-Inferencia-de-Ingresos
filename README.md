# Proyecto Inferencia de Ingresos
<br/>
<p align="center">
 <a href="https://plataformavirtual.itla.edu.do/pluginfile.php/929260/mod_assign/introattachment/0/Descripci%C3%B3n%20Proyecto%20Inferencia%20de%20Ingresos.pdf?forcedownload=1" target="_blank" style="color:blue; font-size:20px; text-decoration:none; font-weight: bold;">
    Descarga el archivo PDF con los requerimientos aquí
 </a>
</p>


## INSTRUCCIONES

En este proyecto desarrollarán un modelo de inferencia de ingresos usando datos de la nómina pública del estado, completando el pipeline de data science explicado en clase y comparando distintos métodos de preparación de datos, EDA, modelamiento y validación.

## I. Resumen

El objetivo de este proyecto es desarrollar un modelo de inferencia de ingresos que estime el ingreso de una persona en base a sus características usando un modelo de Machine Learning supervisado para regresión. Para esto se entrenarán distintos modelos de regresión, comparando su performance, hiperparámetros para finalmente seleccionar el mejor de todos y poder hacer predicciones con el modelo. Esto se realizará empleando la librería scikit-learn con Python.

## II. Detalles de Implementación

El pipeline de un proyecto de Machine Learning supervisado consiste en lo siguiente:

1. **Recolección de la data**: Importar el dataset crudo.
2. **Preparación de la data/ preprocesamiento la data**:
   - Estandarizar formatos y homogenizar datos.
   - Ingeniería de características (eliminación de características redundantes o innecesarias).
   - Limpieza de filas nulas, vacías o con error.
   - Aplicar un encoder o codificador a las características no numéricas (Guardar diccionario de codificación).
   - Normalizar y estandarizar la data con un escalador de datos (Convertirlos en datos con media cero y desviación estándar uno).
3. **Análisis descriptivo de la data (EDA)**:
   - Analizar la data con gráficas.
   - Interpretar las estadísticas de los datos.
   - Interpretar patrones de los datos con consultas y métodos de visualización.
4. **Entrenamiento del modelo**:
   - División del dataset en entradas y salidas/etiquetas (x, y).
   - División del dataset en entrenamiento y testeo.
   - Entrenamiento de cada algoritmo con el dataset.
5. **Validación y testeo del modelo**:
   - Análisis de performance (matriz de confusión, precisión, recall, accuracy, etc).
   - Selección de algoritmo óptimo.
6. **Despliegue del modelo y comprobación con data recién creada**:
   - Conversión de data nueva cruda a formato de entrada del algoritmo (codificación, escalado, etc).
   - Predicción de categoría del dato.

La librería scikit-learn con el apoyo de pandas, numpy y matplotlib cuenta con múltiples módulos para realizar estas funciones de forma sencilla en Python.

## III. Requerimientos de la Entrega

### Recopilación y Preparación de Datos

- Deben recopilar mínimo cinco nóminas distintas de instituciones diferentes.
- El dataset debe tener al menos 5000 filas después de la limpieza.
- Debe tener al menos 6 características de entrada y una etiqueta de salida después de la limpieza.
- La etiqueta de salida debe ser de tipo real.
- Debe haber al menos una característica de entrada tipo entero, una tipo decimal y una tipo categórica.
- Deben cargar estas nóminas en python usando pandas.
- Deben concatenar estas nóminas en una sola, formatearlos y homogenizarlas usando pandas.
- Deben realizar un análisis de la calidad de datos (cantidad de celdas nulas, cantidad de celdas mal formateadas, entre otros).
- Obtener estadísticas básicas de las columnas (promedio, mediana, mínimo, máximo, desviación estándar).
- Ver las distribuciones de los datos de cada columna.
- Analizar estas distribuciones en conjunto con otras variables (dist. de ingreso por género, por cargo, por institución, entre otros).
- Identificar correlaciones entre las variables.

### Modelamiento

- Debe entrenar los siguientes modelos de regresión con el dataset:
  1. Ordinary Least Squares Regression
  2. Ridge Regression
  3. Bayesian Regression
  4. Lasso Regression
  5. Nearest Neighbors Regression
  6. Decision Tree Regression
  7. Random Forest Regression
  8. SVM (Support Vector Machine) Regression
  9. Neural Network MLP Regression
  10. Ada Boost Regressor
- Debe mostrar las métricas de rendimiento de cada uno de los modelos.
- Se debe poder hacerle pruebas a cada uno de los modelos ingresando un archivo “.csv”.
- Las fuentes deben estar debidamente documentadas con docstrings y anotaciones.

### Entrega

- Debe entregar un cuaderno de Jupyter con el código fuente con el se entrenaron los modelos y se hicieron las pruebas.
- Debe entregar un documento de infraestructura del proyecto que contenga:
 - Gráficas analíticas de modelo (EDA).
 - Proporción de testeo/entrenamiento.
 - Descripción de algoritmos empleados.
 - Hiperparámetros usados en los modelos.
 - Estadísticas de modelos.
 - Explicación de cómo funciona el sistema de regresión creado.
 - Importancia de características.
