#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from pmdarima.arima import ARIMA
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")  # Suppress ARIMA warnings


# Load data from CSV
defasado_com_cgpj = './CGP.csv'
defasado_com_juros = './txjAA.csv'

logar_previsoes = False
logar_feature_importance = True
media_movel_4_meses = 0
soma_4_meses = 1

valores = []

with open(defasado_com_cgpj, 'r') as arquivo_cru:
    arquivo_lido = csv.reader(arquivo_cru)  

    # Skip the header row
    next(arquivo_lido)

    for linha in arquivo_lido:
        valores.append([float(val) for val in linha])

valores = np.array(valores)

# Separate features and target variables
X = valores[:, 2:]
y = valores[:, media_movel_4_meses]

# Split data into learning and test periods
learning_end_index = 153 + 12 + 6  # Index for June 2020
X_learn, X_test = X[:learning_end_index], X[learning_end_index:]
y_learn, y_test = y[:learning_end_index], y[learning_end_index:]

# Normalize data using StandardScaler
scaler = StandardScaler()
X_learn_scaled = scaler.fit_transform(X_learn)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------

print("--------------- BAGGING -------------\n")

bagging_regressor = BaggingRegressor(n_estimators=100, random_state=42)
bagging_regressor.fit(X_learn_scaled, y_learn)
y_pred_bagging = bagging_regressor.predict(X_test_scaled)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)

base_regressors = bagging_regressor.estimators_

# Initialize a list to store feature importances from all base regressors
feature_importances = []

# Iterate through each base regressor (decision tree) to extract feature importances
for base_regressor in base_regressors:
    if hasattr(base_regressor, 'feature_importances_'):
        # Some base regressors may not have feature importances, so check if the attribute exists
        feature_importances.append(base_regressor.feature_importances_)

# Calculate the average feature importances across all base regressors
average_feature_importances = np.mean(feature_importances, axis=0)

if (logar_feature_importance):
    for valor in average_feature_importances:
        print(f"{valor}".replace(".", ","))

if (logar_previsoes):
    for valor in y_pred_bagging:
        print(f"{valor}".replace(".", ","))

print("\n")
print(f'RAIZ QUADRADA DE ERRO DO BAGGING: {mse_bagging}\n')

# -----------------------------------------------------------

print("-------------- RANDOM FOREST --------------\n")

random_forest_regressor = RandomForestRegressor(
    n_estimators=100, random_state=42)
random_forest_regressor.fit(X_learn_scaled, y_learn)
y_pred_rf = random_forest_regressor.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)

if (logar_feature_importance):
    for valor in random_forest_regressor.feature_importances_:
        print(f"{valor}".replace(".", ","))

if (logar_previsoes):
    for valor in y_pred_rf:
        print(f"{valor}".replace(".", ","))
    
print("\n")
print(f'RAIZ QUADRADA DE ERRO DO RANDOM FOREST: {mse_rf}\n')

# ----------------------------------------------------------

print("--------------- XGBOOST ------------\n")

xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)
xgb_regressor.fit(X_learn_scaled, y_learn)
y_pred_xgb = xgb_regressor.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

if (logar_feature_importance):
    for valor in xgb_regressor.feature_importances_:
        print(f"{valor:.20f}".replace(".", ","))


if (logar_previsoes):
    for valor in y_pred_xgb:
        print(f"{valor}".replace(".", ","))
    
print("\n")
print(f'RAIZ QUADRADA DE ERRO DO XGBOOST: {mse_xgb}\n')

# ----------------------------------------------------------

print("--------------- ARIMA (1,1,1) ------------\n")

arima_model = ARIMA(order=(1, 1, 1))
arima_model_fit = arima_model.fit(y_learn)
y_pred_arima = arima_model_fit.predict(n_periods=len(y_test))
mse_arima = mean_squared_error(y_test, y_pred_arima)


if (logar_previsoes):
    for valor in y_pred_arima:
        print(f"{valor}".replace(".", ","))
  
print("\n")
print(f'RAIZ QUADRADA DE ERRO DO ARIMA: {mse_arima}\n')
