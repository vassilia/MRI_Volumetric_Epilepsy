import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Read the merged data from Excel into a DataFrame
merged_data = pd.read_excel('merged_data.xlsx')

## Missing values
# Check for missing values in the specific columns
missing_values = merged_data[['volume (ml)', '% of eTIV']].isnull().sum()
print("Missing Values:\n", missing_values)

# Fill missing values with mean or other appropriate values
merged_data['volume (ml)'] = merged_data['volume (ml)'].fillna(merged_data['volume (ml)'].mean())
merged_data['% of eTIV'] = merged_data['% of eTIV'].fillna(merged_data['% of eTIV'].mean())

# Verify if missing values are handled
missing_values_after = merged_data[['volume (ml)', '% of eTIV']].isnull().sum()
print("\nMissing Values After Handling:\n", missing_values_after)

## Encoding
# Encode categorical variables using label encoding
label_encoder = LabelEncoder()
merged_data['sex_encoded'] = label_encoder.fit_transform(merged_data['sex'])
merged_data['diagnosis_encoded'] = label_encoder.fit_transform(merged_data['diagnosis'])
merged_data['MRI_results_encoded'] = label_encoder.fit_transform(merged_data['MRI'])
merged_data['EEG_results_encoded'] = label_encoder.fit_transform(merged_data['EEG'])

# Identify the features (X) and target variable (y)
features = ['volume (ml)', '% of eTIV', 'sex_encoded', 'MRI_results_encoded', 'EEG_results_encoded']
target = 'diagnosis_encoded'

X = merged_data[features]
y = merged_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Decision Tree model
model = DecisionTreeClassifier()

# Train the model using the training data
model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# Compare the predicted labels with the actual labels
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)
