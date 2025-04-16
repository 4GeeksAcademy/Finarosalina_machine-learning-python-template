# Your code here
import requests

# URL del archivo CSV
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"

# Hacer una solicitud GET para obtener el archivo
response = requests.get(url)

# Verificar si la solicitud fue exitosa (código 200)
if response.status_code == 200:
    # Guardar el contenido del archivo en la carpeta data/raw
    with open('/workspaces/Finarosalina_machine-learning-python-template/data/raw/bank-marketing-campaign-data.csv', 'wb') as file:
        file.write(response.content)
    print("Archivo descargado correctamente!")
else:
    print(f"Hubo un problema al descargar el archivo: {response.status_code}")


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns



df=pd.read_csv('/workspaces/Finarosalina_machine-learning-python-template/data/raw/bank-marketing-campaign-data.csv',  sep=';')
df.head()
# shape  (41188, 21)


# Verificar si hay filas duplicadas
duplicados = df.duplicated()
print(df[duplicados])

# Veo que hay 12 filas duplicadas, por lo que las elimino
print(f"Total de filas duplicadas: {duplicados.sum()}")

df=df.drop_duplicates()
duplicados = df.duplicated()
print(f"Total de filas duplicadas: {duplicados.sum()}")

fig, axis = plt.subplots(4, 3, figsize=(12, 10))

# Crear un histograma múltiple
sns.histplot(ax=axis[0, 0], data=df, x="y").set_xlim(-0.1, 1.1)
sns.histplot(ax=axis[0, 1], data=df, x="job").set(ylabel=None)
sns.histplot(ax=axis[0, 2], data=df, x="marital").set(ylabel=None)
sns.histplot(ax=axis[1, 0], data=df, x="education")
sns.histplot(ax=axis[1, 1], data=df, x="default").set(ylabel=None)
sns.histplot(ax=axis[1, 2], data=df, x="housing").set(ylabel=None)
sns.histplot(ax=axis[2, 0], data=df, x="loan").set(ylabel=None)
sns.histplot(ax=axis[2, 1], data=df, x="contact").set(ylabel=None)
sns.histplot(ax=axis[2, 2], data=df, x="month").set(ylabel=None)
sns.histplot(ax=axis[3, 0], data=df, x="day_of_week").set(ylabel=None)
sns.histplot(ax=axis[3, 1], data=df, x="poutcome").set(ylabel=None)

# Eliminar el subplot vacío que sobra (axis[3, 2])
fig.delaxes(axis[3, 2])

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()


df['job'].value_counts().head(10)
df['job'].value_counts().head(10).plot(kind='barh', figsize=(8, 5))
plt.xlabel('Frecuencia')
plt.title('Top 10 trabajos más frecuentes')
plt.gca().invert_yaxis()  #  para que el más frecuente quede arriba
plt.show()

from matplotlib.gridspec import GridSpec

# Crear la figura y el GridSpec
fig = plt.figure(figsize=(15, 30))

# Usar GridSpec para dividir el espacio (10 filas, 2 columnas)
gs = GridSpec(10, 2, figure=fig, height_ratios=[6, 1] * 5)  # 5 pares de histograma y boxplot (10 filas en total)

# Crear los gráficos de histograma y boxplot
sns.histplot(ax=fig.add_subplot(gs[0, 0]), data=df, x="age").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[1, 0]), data=df, x="age")

sns.histplot(ax=fig.add_subplot(gs[0, 1]), data=df, x="duration").set(xlabel=None, ylabel=None)
sns.boxplot(ax=fig.add_subplot(gs[1, 1]), data=df, x="duration")

sns.histplot(ax=fig.add_subplot(gs[2, 0]), data=df, x="campaign").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[3, 0]), data=df, x="campaign")

sns.histplot(ax=fig.add_subplot(gs[2, 1]), data=df, x="pdays").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[3, 1]), data=df, x="pdays")

sns.histplot(ax=fig.add_subplot(gs[4, 0]), data=df, x="previous").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[5, 0]), data=df, x="previous")

sns.histplot(ax=fig.add_subplot(gs[4, 1]), data=df, x="emp.var.rate").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[5, 1]), data=df, x="emp.var.rate")

sns.histplot(ax=fig.add_subplot(gs[6, 0]), data=df, x="cons.price.idx").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[7, 0]), data=df, x="cons.price.idx")

sns.histplot(ax=fig.add_subplot(gs[6, 1]), data=df, x="cons.conf.idx").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[7, 1]), data=df, x="cons.conf.idx")

sns.histplot(ax=fig.add_subplot(gs[8, 0]), data=df, x="euribor3m").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[9, 0]), data=df, x="euribor3m")

sns.histplot(ax=fig.add_subplot(gs[8, 1]), data=df, x="nr.employed").set(xlabel=None)
sns.boxplot(ax=fig.add_subplot(gs[9, 1]), data=df, x="nr.employed")

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()



df[df['age'] > 70].shape


print(df[df['age'] > 70]['y'].sum())  # son valores atipicos pero aqui el % contratacion se dispara, por lo que no los saco.
porcentaje_contratacion_mayores_70=130/421*100
porcentaje_contratacion_mayores_70   #  30.8 % lo que es considerablemente mayor que el promedio (11,27%)

# df[df['pdays']<100].value_counts()  # 1515

df['y'].describe()

df["y"] = df["y"].map({"yes": 1, "no": 0})

fig, axis = plt.subplots(10, 2, figsize = (15, 30))

# Crear un diagrama de dispersión múltiple
sns.regplot(ax = axis[0, 0], data = df, x = "age", y = "y")
sns.heatmap(df[["y", "age"]].corr(), annot = True, ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = df, x = "duration", y = "y").set(ylabel=None)
sns.heatmap(df[["y", "duration"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = df, x = "campaign", y = "y").set(ylabel=None)
sns.heatmap(df[["y", "campaign"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0])

sns.regplot(ax = axis[2, 1], data = df, x = "pdays", y = "y")
sns.heatmap(df[["y", "pdays"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 1], cbar = False)

sns.regplot(ax = axis[4, 0], data = df, x = "previous", y = "y")
sns.heatmap(df[["y", "previous"]].corr(), annot = True, fmt = ".2f", ax = axis[5,0], cbar = False)

sns.regplot(ax = axis[4, 1], data = df, x = "emp.var.rate", y = "y")
sns.heatmap(df[["y", "emp.var.rate"]].corr(), annot = True, fmt = ".2f", ax = axis[5, 1], cbar = False)

sns.regplot(ax = axis[6, 0], data = df, x = "cons.price.idx", y = "y")
sns.heatmap(df[["y", "cons.price.idx"]].corr(), annot = True, fmt = ".2f", ax = axis[7, 0], cbar = False)

sns.regplot(ax = axis[6, 1], data = df, x = "cons.conf.idx", y = "y")
sns.heatmap(df[["y", "cons.conf.idx"]].corr(), annot = True, fmt = ".2f", ax = axis[7, 1], cbar = False)

sns.regplot(ax = axis[8, 0], data = df, x = "euribor3m", y = "y")
sns.heatmap(df[["y", "euribor3m"]].corr(), annot = True, fmt = ".2f", ax = axis[9, 0], cbar = False)

sns.regplot(ax = axis[8, 1], data = df, x = "nr.employed", y = "y")
sns.heatmap(df[["y", "nr.employed"]].corr(), annot = True, fmt = ".2f", ax = axis[9, 1], cbar = False)


# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

fig, axis = plt.subplots(5, 2, figsize = (25, 40))

sns.countplot(ax = axis[0, 0], data = df, x = "job", hue = "y")
sns.countplot(ax = axis[0, 1], data = df, x = "marital", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = df, x = "education", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[1, 1], data = df, x = "default", hue = "y")
sns.countplot(ax = axis[2, 0], data = df, x = "housing", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[2, 1], data = df, x = "loan", hue = "y")
sns.countplot(ax = axis[3, 0], data = df, x = "contact", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[3, 1], data = df, x = "month", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[4, 0], data = df, x = "day_of_week", hue = "y")
sns.countplot(ax = axis[4, 1], data = df, x = "poutcome", hue = "y").set(ylabel = None)

plt.tight_layout()
plt.show()

fig, axis = plt.subplots(figsize = (10, 5), ncols = 2)

sns.barplot(ax = axis[0], data = df, x = "marital", y = "y", hue = "default")
sns.barplot(ax = axis[1], data = df, x = "poutcome", y = "y", hue = "housing").set(ylabel = None)

plt.tight_layout()

plt.show()

df['marital_n'] = pd.factorize(df['marital'])[0]
df['day_of_week_n'] = pd.factorize(df['day_of_week'])[0]
df['poutcome_n'] = pd.factorize(df['poutcome'])[0]
df['contact_n'] = pd.factorize(df['contact'])[0]
df['month_n'] = pd.factorize(df['month'])[0]
df['housing_n'] = pd.factorize(df['housing'])[0]
df['loan_n'] = pd.factorize(df['loan'])[0]
df['education_n'] = pd.factorize(df['education'])[0]
df['default_n'] = pd.factorize(df['default'])[0]
df['job_n'] = pd.factorize(df['job'])[0]


fig, axis = plt.subplots(figsize=(20, 15))

sns.heatmap(df[['job_n', 'marital_n', 'education_n', 'default_n', 'housing_n', 
                'loan_n', 'contact_n', 'month_n', 'day_of_week_n', 'poutcome_n', 'y']].corr(),
            annot=True, fmt=".2f", cmap="coolwarm")

plt.title("Matriz de correlación entre variables categóricas codificadas")
plt.tight_layout()
plt.show()


fig, axis = plt.subplots(figsize = (15, 10))

sns.heatmap(df[['job_n', 'marital_n', 'education_n', 'default_n', 'housing_n', 
                'loan_n', 'contact_n', 'month_n', 'day_of_week_n', 'poutcome_n', 'y', 'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()



# los atributos que no muestran corelación son; age, job, marital, education, housing, day of week y cons.conf.idx
df_reducido = df.drop([
    'age', 'job', 'job_n', 'marital_n', 'marital', 'education', 'education_n',
    'housing', 'housing_n', 'day_of_week', 'day_of_week_n', 'cons.conf.idx'
], axis=1).copy()


df_reducido.drop(['loan', 'month', 'default', 'poutcome', 'contact', 'default'], axis=1, inplace=True)

# Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
Q1 = df_reducido['duration'].quantile(0.25)
Q3 = df_reducido['duration'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites inferior y superior
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(IQR, 2)}")

# Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
Q1 = df_reducido['duration'].quantile(0.25)
Q3 = df_reducido['duration'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites inferior y superior
lower_limit = Q1 - 3 * IQR
upper_limit = Q3 + 3 * IQR

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(IQR, 2)}")

df_reducido[df_reducido['duration']>970]['y'].sum() # 1043 valores, con 618 contrataciones, me parece que no se pueden descartar tantos.

# Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
Q1 = df_reducido['campaign'].quantile(0.25)
Q3 = df_reducido['campaign'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites inferior y superior
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(IQR, 2)}")

# Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
Q1 = df_reducido['campaign'].quantile(0.25)
Q3 = df_reducido['campaign'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites inferior y superior
lower_limit = Q1 - 3 * IQR
upper_limit = Q3 + 3 * IQR

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(IQR, 2)}")

df_reducido[df_reducido['campaign']>10]['y'].sum()  # 689 valores con 27 resultados positivos
df_reducido=df_reducido[df_reducido['campaign']<10]

df_reducido.columns

from sklearn.model_selection import train_test_split

X = df_reducido.drop('y', axis=1)  
y = df_reducido['y'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head(20)

from sklearn.preprocessing import MinMaxScaler

# Variables que NO se escalan
no_escalar = ['contact_n', 'loan_n', 'default_n']

# Variables que SÍ se escalan (todas menos las binarias)
variables_a_escalar = [col for col in X_train.columns if col not in no_escalar]

# Escalar
scaler = MinMaxScaler()

# Escalar solo las columnas necesarias
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[variables_a_escalar]), columns=variables_a_escalar, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[variables_a_escalar]), columns=variables_a_escalar, index=X_test.index)


X_train_final = pd.concat([X_train_scaled, X_train[no_escalar]], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test[no_escalar]], axis=1)

X_train_final.head()


from sklearn.feature_selection import f_classif, SelectKBest


selection_model = SelectKBest(f_classif, k = 11)
selection_model.fit(X_train_final, y_train)

ix = selection_model.get_support()

X_train_sel = pd.DataFrame(selection_model.transform(X_train_final), columns = X_train_final.columns[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_final), columns = X_test_final.columns[ix])

# Ver los primeros 5 registros del conjunto seleccionado de X_train
X_train_sel.head()


# Guardar X_train_sel y y_train en un archivo CSV
train_data = X_train_sel.copy()
train_data['y'] = y_train  # Añadimos las etiquetas de destino

# Guardar X_test_sel y y_test en un archivo CSV
test_data = X_test_sel.copy()
test_data['y'] = y_test  # Añadimos las etiquetas de destino

# Guardar los DataFrames en archivos CSV
train_data.to_csv("/workspaces/Finarosalina_machine-learning-python-template/data/processed/processed_train.csv", index=False)
test_data.to_csv("/workspaces/Finarosalina_machine-learning-python-template/data/processed/processed_test.csv", index=False)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_sel, y_train)
# Realizar predicciones
y_pred = model.predict(X_test_sel)

# Evaluar el modelo
from sklearn.metrics import accuracy_score, confusion_matrix

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión: {accuracy}")


from sklearn.metrics import confusion_matrix

# Confusion matrix para df_reducido
cm = confusion_matrix(y_test, y_pred)

labels = sorted(y_test.unique())  # por ejemplo: ['no', 'yes']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Visualización
plt.figure(figsize=(4, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")

plt.tight_layout()
plt.show()


coeficientes = pd.Series(model.coef_[0], index=X_train_sel.columns)
print(coeficientes.sort_values(ascending=False))



coef_df = coeficientes.sort_values().plot(kind='barh', figsize=(8, 6), color='teal')
plt.title("Importancia de las variables en la predicción (coeficientes)")
plt.xlabel("Peso del coeficiente")
plt.tight_layout()
plt.show()


# OPTIMIZACION DEL MODELO

# Crear el modelo
model = LogisticRegression(max_iter=1000, random_state=42)

# Entrenar el modelo
model.fit(X_train_sel, y_train)

# Predecir con los datos de prueba
y_pred = model.predict(X_test_sel)

# Calcular accuracy
base_accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {base_accuracy:.4f}")

import nbformat

# Cargar el notebook
with open('/workspaces/Finarosalina_machine-learning-python-template/src/explore.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Extraer solo el código
code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
code_content = '\n\n'.join(code_cells)

# Guardar como .py
with open('/workspaces/Finarosalina_machine-learning-python-template/src/app.py', 'w', encoding='utf-8') as f:
    f.write(code_content)
