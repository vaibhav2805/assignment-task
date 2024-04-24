import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the preprocessed data from pre_processed_data.csv
data = pd.read_csv("pre_processed_data.csv")

# Extract input features (X) and target variable (y)
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, 
                                   max_depth=None, 
                                   min_samples_split=2, 
                                   min_samples_leaf=1, 
                                   max_features=25, 
                                   bootstrap=True, 
                                   criterion='entropy', 
                                   random_state=42)

# Train the Random Forest classifier
rf_model.fit(X_train, y_train)

# Evaluate the Random Forest classifier on test data
accuracy = rf_model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy}')