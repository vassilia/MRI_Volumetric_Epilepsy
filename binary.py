import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Read the Excel file
xls = pd.ExcelFile('mridata .xlsx')

# List to store the data for each patient
data_list = []

# Iterate over each sheet in the Excel file
for sheet_name in xls.sheet_names:
    # Read the data for the current sheet
    data = xls.parse(sheet_name)

    # Extract the features (X) and target variable (y)
    X = data.drop('diagnosis', axis=1)  # Assuming 'diagnosis' is the column name for the target variable
    y = data['diagnosis']

    # Create a tuple of features and target variable for the current patient
    patient_data = (X, y)

    # Append the patient data to the list
    data_list.append(patient_data)

# Now you have a list 'data_list' containing tuples of (X, y) for each patient
# You can process each patient's data separately as per your requirements

# Generate dummy data
X = np.random.randn(20, 100)  # 20 samples with 100 features
y = np.random.randint(2, size=20)  # Binary labels (0 or 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
