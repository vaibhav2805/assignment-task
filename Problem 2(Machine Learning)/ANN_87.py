import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the preprocessed data from pre_processed_data.csv
data = pd.read_csv("pre_processed_data.csv")

# Extract input features (X) and target variable (y)
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(75, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(50, activation='tanh'),
    Dense(50, activation='tanh'),
    Dense(50, activation='tanh'),
    Dense(50, activation='tanh'),
    Dense(50, activation='tanh'),
    Dense(3, activation='softmax')  # Output layer with 3 neurons for 3 classes and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=200, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')