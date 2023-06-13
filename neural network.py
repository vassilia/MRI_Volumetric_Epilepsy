import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report

# Read the merged data from Excel into a DataFrame
merged_data = pd.read_excel('merged_data.xlsx')

## Missing Values ##
# Check for missing values in the specific columns
missing_values = merged_data[['volume (ml)', '% of eTIV']].isnull().sum()
print("Missing Values:\n", missing_values)

# Fill missing values with mean or other appropriate values
merged_data['volume (ml)'] = merged_data['volume (ml)'].fillna(merged_data['volume (ml)'].mean())
merged_data['% of eTIV'] = merged_data['% of eTIV'].fillna(merged_data['% of eTIV'].mean())

# Verify if missing values are handled
missing_values_after = merged_data[['volume (ml)', '% of eTIV']].isnull().sum()
print("\nMissing Values After Handling:\n", missing_values_after)

## Encoding ##
# Encode categorical variables using label encoding
label_encoder = LabelEncoder()
merged_data['sex_encoded'] = label_encoder.fit_transform(merged_data['sex'])
merged_data['diagnosis_encoded'] = label_encoder.fit_transform(merged_data['diagnosis'])

# Identify the features (X) and target variable (y)
features = ['volume (ml)', '% of eTIV', 'sex_encoded']
target = 'diagnosis_encoded'

X = merged_data[features]
y = merged_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(features),)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the testing data
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", accuracy)

# Use the trained model to make predictions on the testing data
y_pred = model.predict_classes(X_test)

# Convert the encoded predictions back to original labels
predicted_labels = label_encoder.inverse_transform(y_pred)

# Compare the predicted labels with the actual labels
report = classification_report(y_test, predicted_labels)

# Print the classification report
print(report)
