import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Read the merged data from Excel into a DataFrame
merged_data = pd.read_excel('merged_data.xlsx')

## missing values
# Check for missing values in the specific columns
missing_values = merged_data[['volume (ml)', '% of eTIV']].isnull().sum()
print("Missing Values:\n", missing_values)

# Fill missing values with mean or other appropriate values
merged_data['volume (ml)'] = merged_data['volume (ml)'].fillna(merged_data['volume (ml)'].mean())
merged_data['% of eTIV'] = merged_data['% of eTIV'].fillna(merged_data['% of eTIV'].mean())

# Verify if missing values are handled
missing_values_after = merged_data[['volume (ml)', '% of eTIV']].isnull().sum()
print("\nMissing Values After Handling:\n", missing_values_after)

##encoding

# Encode categorical variables using one-hot encoding
categorical_columns = ['structure', 'type', 'hemisphere', 'diagnosis', 'sex', 'MRI', 'EEG']
encoded_data = pd.get_dummies(merged_data, columns=categorical_columns)

# Identify the features (X) and target variable (y)
features = ['volume (ml)', '% of eTIV', 'sex_Female', 'sex_Male', 'MRI Results_Negative', 'MRI Results_Positive',
            'EEG Results_Negative', 'EEG Results_Positive']
target = 'diagnosis'

X = encoded_data[features]
y = encoded_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Logistic Regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# Compare the predicted labels with the actual labels
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)
