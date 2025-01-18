#Problem klasyfikacji z wykorzystaniem zwykłej sieci warstwowej w Pythonie

# Ten kod został stworzony, aby rozwiązać problem klasyfikacji przy użyciu sieci neuronowej. 
# Zadaniem modelu jest nauczenie się rozpoznawania trzech klas kwiatów ze zbioru danych Iris na podstawie podanych cech (np. długość płatków i działek). 
# Kod pokazuje proces treningu, oceny i interpretacji wyników działania modelu.

# Importujemy potrzebne biblioteki
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris

# Wczytanie danych Iris
data = load_iris()
X = data.data  # Dane wejściowe (cechy)
y = data.target  # Klasy wyjściowe

# One-hot encoding dla klas (potrzebne dla sieci neuronowej)
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Podział danych na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie modelu sieci neuronowej
model = Sequential([
    Dense(16, input_dim=X.shape[1], activation='relu'),  # Pierwsza warstwa ukryta
    Dense(8, activation='relu'),  # Druga warstwa ukryta
    Dense(3, activation='softmax')  # Warstwa wyjściowa (dla 3 klas)
])

# Kompilacja modelu
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

# Ocena na zbiorze testowym
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Dokładność modelu na zbiorze testowym: {accuracy:.2f}")

# Prognozy i tablica błędów
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Klasy przewidywane
y_test_classes = np.argmax(y_test, axis=1)  # Klasy rzeczywiste

# Tablica błędów
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=data.target_names)

# Wizualizacja tablicy błędów
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Raport klasyfikacji
print("Raport klasyfikacji:")
print(classification_report(y_test_classes, y_pred_classes, target_names=data.target_names))

#Wyjaśnienie kroków
#Wczytanie danych: Korzystamy z zestawu Iris, który zawiera dane o trzech gatunkach kwiatów (trzy klasy).
#Przygotowanie danych:
#Standaryzacja nie jest wymagana dla prostego modelu, ponieważ dane są już w odpowiedniej skali.
#Wykorzystujemy one-hot encoding dla zmiennych docelowych, co jest wymagane w przypadku categorical_crossentropy.
#Budowa sieci neuronowej:
#Prosta sieć z 2 warstwami ukrytymi (16 i 8 neuronów) oraz funkcją aktywacji relu.
#Warstwa wyjściowa z funkcją aktywacji softmax do klasyfikacji wieloklasowej.
#Trenowanie modelu: Używamy optymalizatora Adam i trenowanie przez 50 epok.
#Ocena i wizualizacja wyników:
#Wyliczamy dokładność modelu.
#Tworzymy tablicę błędów i raport klasyfikacji, aby zrozumieć, które klasy są najlepiej (lub najsłabiej) rozróżniane.
#Wyniki
#Tablica błędów: Pokazuje, ile razy model prawidłowo (lub błędnie) przewidział daną klasę.
#Raport klasyfikacji: Podaje metryki takie jak precyzja, czułość i F1-score dla każdej klasy.

#Interpretacja
##Model działa bardzo dobrze dla klasy "setosa", ale ma pewne trudności w odróżnianiu "versicolor" od "virginica".
##Wysoka dokładność (90%) świadczy o solidnej jakości modelu.
##Poprawa modelu może obejmować:
##Dostosowanie architektury sieci neuronowej,
##Lepsze przygotowanie danych (np. skalowanie cech),
##Trenowanie modelu przez więcej epok.