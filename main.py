import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error




# PRE-PROCESAMIENTO DE NOMINA DE TURISMO(TURISMO)
turismo_df = pd.read_csv('Nominas/nomina_turismo.csv')
turismo_df = turismo_df.iloc[:, :-1] # Eliminar la última columna que no tiene datos

# Eliminar columnas no necesarias
colums_to_drop = ['NO.', 'NOMBRE', 'GRUPO OCUPACIONAL', 'INICIO CONTRATO', 'Otros Descuentos', 'Sueldo Neto', 'Total Descuentos']
turismo_df.drop(columns=colums_to_drop, inplace=True)

# Renombrar columnas
column_mapping = {
    'SEXO': 'GENERO',
    'CARGO': 'FUNCION',
    'UNIDAD': 'DEPARTAMENTO',
    'SALARIO RD$': 'SUELDO_BRUTO',
    'AFP': 'AFP',
    'Impuesto Sobre Renta ISR': 'ISR',
    'Seguro Familiar Salud SFS': 'SFS',
}

turismo_df.rename(columns=column_mapping, inplace=True)

# Eliminar caracteres no deseados
turismo_df['SUELDO_BRUTO'] = turismo_df['SUELDO_BRUTO'].str.replace(',', '')
turismo_df['AFP'] = turismo_df['AFP'].str.replace(',', '')
turismo_df['ISR'] = turismo_df['ISR'].str.replace(',', '')
turismo_df['SFS'] = turismo_df['SFS'].str.replace(',', '')

# Convertir columnas a tipo numérico
num_cols = ['SUELDO_BRUTO', 'AFP', 'ISR', 'SFS']
turismo_df[num_cols] = turismo_df[num_cols].apply(pd.to_numeric)

turismo_df.fillna(0, inplace=True)

# Agregar la columna INSTITUCION
turismo_df['INSTITUCION'] = 'Ministerio de Turismo'

# Calcular el sueldo neto y agregarlo al dataframe
turismo_df['SUELDO_NETO'] = turismo_df['SUELDO_BRUTO'] - turismo_df['AFP'] - turismo_df['SFS'] - turismo_df['ISR']

# PRE-PROCESAMIENTO DE NOMINA DE OBRAS PUBLICAS(MOPC)

mopc_df = pd.read_csv('Nominas/nomina_mopc.csv')
mopc_df.dropna(inplace=True)

mopc_df['INSTITUCION'] = 'MINISTERIO DE OBRAS PUBLICAS'

columns_to_drop = ['Empleado', 'Tipo de Empleado', 'Otros descuentos', 'Carrera Adm', 'Tipo de Empleado/Cargo', 'Ingreso Neto']
mopc_df.drop(columns=columns_to_drop, inplace=True)

column_mapping = {
    'Cargo': 'FUNCION',
    'Departamento': 'DEPARTAMENTO',
    'Genero': 'GENERO',
    'Descuento AFP': 'AFP',
    'Descuento ISR': 'ISR',
    'Descuento SFS': 'SFS',
    ' Ingreso Bruto ': 'SUELDO_BRUTO'
}

mopc_df.rename(columns=column_mapping, inplace=True)

mopc_df['SUELDO_BRUTO'] = mopc_df['SUELDO_BRUTO'].str.replace(',', '')
mopc_df['ISR'] = mopc_df['ISR'].str.replace(',', '')
mopc_df['AFP'] = mopc_df['AFP'].str.replace(',', '')
mopc_df['SFS'] = mopc_df['SFS'].str.replace(',', '')

mopc_df['SUELDO_BRUTO'] = pd.to_numeric(mopc_df['SUELDO_BRUTO'], errors='coerce')
mopc_df['ISR'] = pd.to_numeric(mopc_df['ISR'], errors='coerce')
mopc_df['AFP'] = pd.to_numeric(mopc_df['AFP'], errors='coerce')
mopc_df['SFS'] = pd.to_numeric(mopc_df['SFS'], errors='coerce')

mopc_df.fillna(0, inplace=True)

mopc_df['SUELDO_NETO'] = mopc_df['SUELDO_BRUTO'] - mopc_df['AFP'] - mopc_df['SFS'] - mopc_df['ISR']

# PRE-PROCESAMIENTO DE NOMINA DE MIGRACION(DGM)

dgm_df = pd.read_csv('Nominas/nomina_dgm.csv')
dgm_df = dgm_df.iloc[:, :-2]

column_names_list = dgm_df.columns.tolist()

new_column_names = [column.strip().replace(' ', '_') for column in column_names_list]
rename_dict = {old_name: new_name for old_name, new_name in zip(column_names_list, new_column_names)}

dgm_df.rename(columns=rename_dict, inplace=True)
dgm_df.rename(columns={'SUELDO_BRUTO_(RD$)': 'SUELDO_BRUTO'}, inplace=True)

dgm_df['GENERO'] = dgm_df['GENERO'].str.upper()

dgm_df['SUELDO_BRUTO'] = dgm_df['SUELDO_BRUTO'].str.replace(',', '')
dgm_df['ISR'] = dgm_df['ISR'].str.replace(',', '')
dgm_df['AFP'] = dgm_df['AFP'].str.replace(',', '')
dgm_df['SFS'] = dgm_df['SFS'].str.replace(',', '')

dgm_df['SUELDO_BRUTO'] = pd.to_numeric(dgm_df['SUELDO_BRUTO'], errors='coerce')
dgm_df['ISR'] = pd.to_numeric(dgm_df['ISR'], errors='coerce')
dgm_df['AFP'] = pd.to_numeric(dgm_df['AFP'], errors='coerce')
dgm_df['SFS'] = pd.to_numeric(dgm_df['SFS'], errors='coerce')

dgm_df.fillna(0, inplace=True)

dgm_df['SUELDO_NETO'] = dgm_df['SUELDO_BRUTO'] - dgm_df['AFP'] - dgm_df['SFS'] - dgm_df['ISR']

dgm_df['INSTITUCION'] = 'DIRECCION GENERAL DE MIGRACION'

non_essential_columns = ['NOMBRE', 'ESTATUS', 'TOTAL_DESC.', 'NETO', 'OTROS_DESC.']
dgm_df = dgm_df.drop(columns=non_essential_columns)

# PRE-PROCESAMIENTO DE NOMINA DEL MINISTERIO DE CULTURA

cultura_df = pd.read_csv('Nominas/nomina_cultura.csv')
cultura_df.dropna(inplace=True)

cultura_df['INSTITUCION'] = 'MINISTERIO DE CULTURA'

columns_to_drop = ['NOMBRE Y APELLIDO', 'CATEGORIA DEL SERVIDOR', 'OTROS DESC', 'INGRESO NETO']
cultura_df.drop(columns=columns_to_drop, inplace=True)

column_mapping = {
    'CARGO': 'FUNCION',
    'DIRECCIÓN O DEPARTAMENTO': 'DEPARTAMENTO',
    'GENERO': 'GENERO',
    'AFP': 'AFP',
    'ISR': 'ISR',
    'SFS': 'SFS',
    ' INGRESO BRUTO ': 'SUELDO_BRUTO'
}

cultura_df.rename(columns=column_mapping, inplace=True)

cultura_df['SUELDO_BRUTO'] = cultura_df['SUELDO_BRUTO'].str.replace(',', '')
cultura_df['ISR'] = cultura_df['ISR'].str.replace(',', '')
cultura_df['AFP'] = cultura_df['AFP'].str.replace(',', '')
cultura_df['SFS'] = cultura_df['SFS'].str.replace(',', '')

cultura_df['SUELDO_BRUTO'] = pd.to_numeric(mopc_df['SUELDO_BRUTO'], errors='coerce')
cultura_df['ISR'] = pd.to_numeric(cultura_df['ISR'], errors='coerce')
cultura_df['AFP'] = pd.to_numeric(cultura_df['AFP'], errors='coerce')
cultura_df['SFS'] = pd.to_numeric(cultura_df['SFS'], errors='coerce')

cultura_df.fillna(0, inplace=True)

cultura_df['SUELDO_NETO'] = cultura_df['SUELDO_BRUTO'] - cultura_df['AFP'] - cultura_df['SFS'] - cultura_df['ISR']

print(cultura_df.describe())

# CONCATENANDO DATAFRAMES

final_df = pd.concat([mopc_df, dgm_df, turismo_df, cultura_df])
final_df.reset_index(drop=True, inplace=True)

# Graficos

numeric_columns = dgm_df.select_dtypes(include=['float64', 'int64']).columns

fig, axs = plt.subplots(len(numeric_columns), 3, figsize=(20, 8 * len(numeric_columns)))

for i, column in enumerate(numeric_columns):
    # Histogram
    axs[i, 0].hist(dgm_df[column], bins=20, color='skyblue', edgecolor='black')
    axs[i, 0].set_xlabel(column)
    axs[i, 0].set_ylabel('FRECUENCIA')
    axs[i, 0].set_title(f'HISTOGRAMA DE {column}')

    # KDE plot
    dgm_df[column].plot.kde(ax=axs[i, 1], color='skyblue')
    axs[i, 1].set_xlabel(column)
    axs[i, 1].set_ylabel('DENSIDAD')
    axs[i, 1].set_title(f'ESTIMACION DE DENSIDAD DEL NUCLEO DE {column}')

    # Box plot
    axs[i, 2].boxplot(dgm_df[column])
    axs[i, 2].set_xlabel(column)
    axs[i, 2].set_title(f'DIAGRAMA DE CAJA DE {column}')

    # Print statistics
    statistics = dgm_df[column].describe()
    print(f"\nEstadísticas de {column}:")
    print(statistics)

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

ohe = OneHotEncoder(handle_unknown='ignore')
sts = StandardScaler()

# Ajustar el codificador OneHotEncoder con todas las categorías de los datos completos
cat_feats = final_df.select_dtypes("object").astype(str)
ohe.fit(cat_feats)

features = ["GENERO", "DEPARTAMENTO", "FUNCION", "SUELDO_BRUTO", "AFP", "ISR", "SFS", "INSTITUCION"]
target = "SUELDO_NETO"

# Dividir los datos en conjuntos de entrenamiento/validación y prueba
train_val_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

# Preprocesar los datos de entrenamiento/validación
x_train_val, y_train_val = train_val_df[features], train_val_df[target]
cat_feats_train_val = train_val_df.select_dtypes("object")
num_feats_train_val = train_val_df.select_dtypes(['float64', 'int64']).drop(columns=[target])

cat_feats_encoded_train_val = ohe.transform(cat_feats_train_val).toarray()
num_feats_scaled_train_val = sts.fit_transform(num_feats_train_val)
xp_train_val = np.hstack([cat_feats_encoded_train_val, num_feats_scaled_train_val])

# Preprocesar los datos de prueba utilizando los mismos transformadores
x_test, y_test = test_df[features], test_df[target]
cat_feats_test = test_df.select_dtypes("object")
num_feats_test = test_df.select_dtypes(['float64', 'int64']).drop(columns=[target])

cat_feats_encoded_test = ohe.transform(cat_feats_test).toarray()
num_feats_scaled_test = sts.transform(num_feats_test)
xp_test = np.hstack([cat_feats_encoded_test, num_feats_scaled_test])

models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Bayesian Ridge Regression", BayesianRidge()),
    ("Lasso Regression", Lasso()),
    ("Nearest Neighbors Regression", KNeighborsRegressor()),
    ("Decision Tree Regression", DecisionTreeRegressor()),
    ("Random Forest Regression", RandomForestRegressor()),
    ("SVM Regression", SVR()),
    ("Neural Network MLP Regression", MLPRegressor()),
    ("AdaBoost Regression", AdaBoostRegressor())
]

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
x_train, x_val, y_train, y_val = train_test_split(xp_train_val, y_train_val, test_size=0.25, random_state=42)

model_performance = {}

for name, model in models:
    print(name)
    model.fit(x_train, y_train)
    y_pred_val = model.predict(x_val)
    mae = mean_absolute_error(y_pred_val, y_val)
    mse = mean_squared_error(y_pred_val, y_val)
    model_performance[name] = {'MAE': mae, 'MSE': mse}
    print("MAE:", mae)
    print("MSE:", mse)
    print("")

best_model_name = min(model_performance, key=lambda k: model_performance[k]['MAE'])
best_model = [model for name, model in models if name == best_model_name][0]
print(f"Best Model: {best_model_name}")

# Hacer predicciones en el conjunto de prueba con el mejor modelo
y_pred_test = best_model.predict(xp_test)

# Imprimir predicciones y valores reales
results_df = pd.concat([
    x_test.reset_index(drop=True),
    y_test.reset_index(drop=True),
    pd.Series(y_pred_test, name="Predicted")
], axis=1)

print(results_df[['SUELDO_NETO', 'Predicted']])


