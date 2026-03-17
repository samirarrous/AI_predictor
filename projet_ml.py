import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Création du dataset
data = pd.DataFrame({
    "surface": [50, 60, 70, 80, 90, 100],
    "prix": [150, 180, 210, 240, 270, 300]
})

# 2. Séparer variables
X = data[["surface"]]  # entrée
y = data["prix"]       # sortie

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Créer le modèle
model = LinearRegression()

# 5. Entraîner
model.fit(X_train, y_train)

# 6. Prédire pour une surface de 75m²
prediction = model.predict([[75]])

print("Prix prédit pour 75m² :", prediction)