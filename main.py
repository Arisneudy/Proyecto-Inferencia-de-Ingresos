import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import json

dgm_df = pd.read_csv('Nominas/nomina_dgm.csv')
map_df = pd.read_csv('Nominas/nomina_map.csv')
dgm_df = dgm_df.iloc[:, :-2]

dgm_column_names = dgm_df.columns
column_names_list = dgm_column_names.tolist()

new_column_names = [column.strip().replace(' ', '_') for column in column_names_list]
rename_dict = {old_name: new_name for old_name, new_name in zip(column_names_list, new_column_names)}

dgm_df.rename(columns=rename_dict, inplace=True)
dgm_df.rename(columns={'SUELDO_BRUTO_(RD$)': 'SUELDO_BRUTO'}, inplace=True)

dgm_df['GENERO'] = dgm_df['GENERO'].str.upper()
dgm_df['INSTITUCION'] = 'DIRECCION GENERAL DE MIGRACION'

dgm_df['SUELDO_BRUTO'] = dgm_df['SUELDO_BRUTO'].str.replace(',', '')
dgm_df['ISR'] = dgm_df['ISR'].str.replace(',', '')
dgm_df['AFP'] = dgm_df['AFP'].str.replace(',', '')
dgm_df['SFS'] = dgm_df['SFS'].str.replace(',', '')

dgm_df['SUELDO_BRUTO'] = pd.to_numeric(dgm_df['SUELDO_BRUTO'], errors='coerce')
dgm_df['ISR'] = pd.to_numeric(dgm_df['ISR'], errors='coerce')
dgm_df['AFP'] = pd.to_numeric(dgm_df['AFP'], errors='coerce')
dgm_df['SFS'] = pd.to_numeric(dgm_df['SFS'], errors='coerce')

dgm_df['SUELDO_NETO'] = dgm_df['SUELDO_BRUTO'] - dgm_df['AFP'] - dgm_df['SFS'] - dgm_df['ISR']

non_essential_columns = ['NOMBRE', 'ESTATUS', 'TOTAL_DESC.', 'NETO', 'OTROS_DESC.']
dgm_df = dgm_df.drop(columns=non_essential_columns)

dgm_df.fillna(0, inplace=True)

numeric_columns = dgm_df.select_dtypes(include=['float64', 'int64']).columns

# Increase the figure size here by increasing the width and height
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

    # # Print statistics
    # statistics = dgm_df[column].describe()
    # print(f"\nEstadísticas de {column}:")
    # print(statistics)

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

ohe = OneHotEncoder()
sts = StandardScaler()

encoder_dict = {}
for column in dgm_df.select_dtypes(include=['object']).columns:
    encoded_data = ohe.fit_transform(dgm_df[[column]]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=[f'{column}_{cat}' for cat in ohe.categories_[0]],
                              index=dgm_df.index)
    dgm_df = pd.concat([dgm_df, encoded_df], axis=1)
    encoder_dict[column] = {str(cat): i for i, cat in enumerate(ohe.categories_[0])}

with open('encoder_dict.json', 'w') as f:
    json.dump(encoder_dict, f)

features = [
    "GENERO",
    "DEPARTAMENTO",
    "FUNCION",
    "SUELDO_BRUTO",
    "ISR",
    "AFP",
    "SFS"
]
target = "SUELDO_NETO"

x, y = dgm_df[features], dgm_df[target]

cat_feats = ohe.fit_transform(x.select_dtypes("object")).toarray()
num_feats = sts.fit_transform(x.select_dtypes("number"))

xp = np.hstack([cat_feats, num_feats])

# print(dgm_df)
# print(scaled_dgm_df)

# print(xp)

# print(dgm_df.info())
# print(dgm_df.describe())
# dgm_df["FUNCION"].value_counts().plot.bar()
# print(dgm_df.shape)


