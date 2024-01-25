# Se implementa la regresión linear empleando librerías de Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv("dataset_phishing.csv", sep=",")

df['legitimate'] = (df['status'] == 'legitimate').astype(int)
df['phishing'] = (df['status'] == 'phishing').astype(int)

# Split the data into training and testing sets
X = df[['length_url', 'nb_dots']].values
y = df['phishing'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión
model = LinearRegression()

# Entrenar el modelo 
model.fit(X_train, y_train)

# Hacer predicciones 
y_pred = model.predict(X_test)

# Convertir las predicciones a binario
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy}')

# Graficar el modelo 
plt.scatter(X_test[:, 0], y_test, color='black', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='blue', label='Predicted')
plt.xlabel('length_url')
plt.ylabel('Phishing (1) or Legitimate (0)')
plt.title('Linear Regression Results')
plt.legend()
plt.show()
