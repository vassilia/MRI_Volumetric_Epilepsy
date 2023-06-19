import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Read the merged data from Excel into a DataFrame
merged_data = pd.read_excel('merged_data.xlsx')

## Handling Missing Values

# Fill missing values with mean or other appropriate values
merged_data['volume (ml)'] = merged_data['volume (ml)'].fillna(merged_data['volume (ml)'].mean())
merged_data['% of eTIV'] = merged_data['% of eTIV'].fillna(merged_data['% of eTIV'].mean())

## Encoding Categorical Variables

# Perform one-hot encoding on categorical variables
categorical_columns = ['structure', 'type', 'hemisphere', 'sex', 'MRI', 'EEG']
encoded_data = pd.get_dummies(merged_data, columns=categorical_columns)

# Identify the features (X) and target variable (y)
features = ['volume (ml)', '% of eTIV']  # Update with actual feature columns
target = 'diagnosis'  # Update with actual target column

X = encoded_data[features]
y = encoded_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Gradient Boosting Classifier
model = GradientBoostingClassifier()

# Train the model using the training data
model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# Compare the predicted labels with the actual labels
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)
