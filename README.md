# Este problema representa un escenario común en la industria de viajes, donde las aerolíneas y las agencias de viajes buscan ofrecer a los pasajeros tarifas competitivas y precisas.

Imagina que trabajas en una agencia de viajes en línea que desea ofrecer a los clientes una herramienta que les permita obtener estimaciones precisas del precio de los boletos de avión para su próximo viaje. Sin embargo, el precio de un boleto de avión puede variar significativamente según múltiples factores, como la aerolínea, el destino, la temporada, la clase de servicio y la anticipación de la reserva.

El desafío radica en desarrollar un modelo de Machine Learning que pueda predecir el precio de un boleto de avión en función de estas diversas variables. Esto implica limpiar, explorar y transformar los datos, seleccionar características relevantes, entrenar y afinar modelos predictivos y evaluar su rendimiento de manera precisa.

Considerando los datos entregados en las bases de datos, realiza las siguientes actividades:
# 1. Describe con detenimiento el problema de negocio que se desea resolver, y cómo se va a hacer esto. Indica la metodología, tareas a realizar, variable objetivo a predecir, etc.

Objetivo: Predecir el precio de un boleto de avión usando características del vuelo (como aerolínea, clase, fecha, duración, etc.).

¿Por qué es útil? Ayuda a agencias de viaje a ofrecer tarifas estimadas competitivas y planificar estrategias de precios basadas en predicciones.

Variable Objetivo (target): Price

Metodología:
Exploración y limpieza de datos.
Ingeniería de características.
Selección de variables.
Entrenamiento y evaluación de modelos.
Optimización de hiperparámetros.

# 2. Carga los dataset entregados y genera un reporte de calidad de los mismos. Indica qué estrategias se van a utilizar para aquellos puntos encontrados (Indicar nulos, outliers, valores perdidos, que se hará con esto, etc.)

import pandas as pd

#Se cargan los datos
df_business = pd.read_excel('/Users/nelsonblanco/Desafio_LATAM/Gen99/Machine Learning/prueba FINAL/business.xlsx')

#Revisión general
df_business.info()
df_business.describe()
df_business.isnull().sum()

#Se cargan los datos
df_economy = pd.read_excel('/Users/nelsonblanco/Desafio_LATAM/Gen99/Machine Learning/prueba FINAL/economy.xlsx')

#Revisión general
df_economy.info()
df_economy.describe()
df_economy.isnull().sum()

# 3. Genera un análisis exploratorio de los dataset entregados, un análisis univariado y bivariado. Prioriza los gráficos más importantes y entrega una conclusión a partir de estos.

#Univariado
import seaborn as sns
import matplotlib.pyplot as plt

# gráfico 1
plt.figure(figsize=(8,5))
sns.histplot(df_business['price'], bins=50, kde =True)
plt.title('Distribución de precios')
plt.show()

# gráfico 2
plt.figure(figsize=(8,5))
sns.histplot(df_economy['price'], bins=50, kde =True)
plt.title('Distribución de precios')
plt.show()

# Bivariado
plt.figure(figsize=(10,5))
sns.boxplot(x ='stop', y ='price', data =df_business)
plt.xticks(rotation =90)
plt.title('Precio vs Escalas')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x ='stop', y ='price', data =df_economy)
plt.xticks(rotation =90)
plt.title('Precio vs Escalas')
plt.show()

plt.figure(figsize =(10,5))
sns.barplot(x ='airline', y ='price', data =df_business)
plt.xticks(rotation =90)
plt.title('Precio promedio por aerolínea')
plt.show()

plt.figure(figsize =(10,5))
sns.barplot(x ='airline', y ='price', data =df_economy)
plt.xticks(rotation =90)
plt.title('Precio promedio por aerolínea')
plt.show()

# 4. Realiza un análisis de correlaciones entre las diferentes variables existentes, identificando cuáles son las variables más importantes para la predicción de la variable objetivo.

import seaborn as sns

# matriz de correlación 1
corr_matrix = df_business.select_dtypes(include ='number').corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap ='coolwarm')
plt.title('Matriz de correlación')
plt.show()

# matriz de correlación 2
corr_matrix = df_economy.select_dtypes(include ='number').corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap ='coolwarm')
plt.title('Matriz de correlación')
plt.show()

# 5. Realiza las transformaciones necesarias para realizar el modelamiento posterior y crea las variables que estimes convenientes con ayuda del análisis previo y la expertise del negocio.

def convert_time(t):
    if pd.isnull(t):
        return None
    h, m = 0, 0
    parts = t.lower().split()
    for p in parts:
        if 'h' in p:
            h = int(p.replace('h', ''))
        if 'm' in p:
            m = int(p.replace('m', ''))
    return h * 60 + m

def preprocess_flight_data(df_economy):
    df_economy = df_economy.copy()
    df_economy.drop_duplicates(inplace =True)
    df_economy['date'] = pd.to_datetime(df_economy['date'], errors ='coerce')
    df_economy['month'] = pd.to_numeric(df_economy['month'], errors='coerce')
    df_economy['time_taken_mins'] = df_economy['time_taken'].apply(convert_time)
    df_economy['stop'] = df_economy['stop'].replace({'non-stop': 0, '1-stop': 1, '2+-stop': 2})
    df_economy = pd.get_dummies(df_economy, columns=['airline', 'from', 'to'], drop_first=True)
    df_economy.drop(['date', 'dep_time', 'arr_time', 'ch_code', 'num_code', 'time_taken'], axis=1, inplace=True)
    df_economy.dropna(inplace=True)
    return df_economy

def preprocess_flight_data(df_business):
    df_business = df_business.copy()
    df_business.drop_duplicates(inplace =True)
    df_business['date'] = pd.to_datetime(df_business['date'], errors ='coerce')
    df_business['month'] = pd.to_numeric(df_business['month'], errors='coerce')
    df_business['time_taken_mins'] = df_business['time_taken'].apply(convert_time)
    df_business['stop'] = df_business['stop'].replace({'non-stop': 0, '1-stop': 1, '2+-stop': 2})
    df_business = pd.get_dummies(df_business, columns=['airline', 'from', 'to'], drop_first =True)
    df_business.drop(['date', 'dep_time', 'arr_time', 'ch_code', 'num_code', 'time_taken'], axis =1, inplace =True)
    df_business.dropna(inplace =True)
    return df_business

# 6. Genera una función que encapsule el tratamiento de datos necesario, para entregar un dataset limpio y procesado a partir del dataset original.

economy_clean = preprocess_flight_data(df_economy)
business_clean = preprocess_flight_data(df_business)

# 7. Elige al menos 3 modelos candidatos para resolver el problema. A partir de esto, genera un conjunto de entrenamiento y prueba, para luego entrenar los diferentes modelos.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# economy
X = economy_clean.drop('price', axis =1)
y = economy_clean['price']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state =42)

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'XGBRegressor': XGBRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} score: {model.score(X_test, y_test):.4f}")

# business
X = business_clean.drop('price', axis =1)
y = business_clean['price']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state =42)

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'XGBRegressor': XGBRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} score: {model.score(X_test, y_test):.4f}")

# 8. Elige una grilla de hiperparametros y luego optimízalos, buscando la mejor combinación para cada grilla. Guardar los modelos entrenados.

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

params = {'max_depth': [5, 10], 'n_estimators': [100, 200]}
grid = GridSearchCV(RandomForestRegressor(), params, cv =3, scoring ='neg_mean_squared_error')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# 9. Define al menos 3 métricas para evaluar los modelos entrenados y genera gráficos de comparación. Elige un baseline para ver qué tan buena es tu opción respecto a ese baseline y concluye.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = best_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha =0.5)
plt.xlabel('Actual')
plt.ylabel('Predicho')
plt.title('Actual vs Predicho')
plt.show()

# 10. Genera una conclusión final respecto a qué tan útiles son los resultados encontrados para resolver el problema propuesto y define cuáles podrían ser los próximos pasos para el proyecto.

Conclusión:

Variables importantes: duración, escalas, aerolínea, ruta
XGBoost y RandomForest tuvieron mejor rendimiento
Siguientes pasos: agregar más datos, refinar variables, desplegar modelo
