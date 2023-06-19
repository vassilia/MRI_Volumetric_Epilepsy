from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, Normalizer
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Read the Excel file
excel_file = pd.ExcelFile('structured_data.xlsx')

# Read the structure-related data sheet into a DataFrame
structure_data = excel_file.parse('StructureData')

# Read the additional patient information sheet into a DataFrame
patient_info = excel_file.parse('PatientInfo')

# Merge the two DataFrames based on the 'Patient' column
merged_data = pd.merge(structure_data, patient_info, on='Patient')

# Print the merged data for verification
print(merged_data)

# Save the merged data to an Excel file
merged_data.to_excel('merged_data.xlsx', index=False)

# Read the merged data from Excel into a DataFrame
merged_data = pd.read_excel('merged_data.xlsx')

## Scaling ##
# Perform Min-Max scaling on the 'volume (ml)' feature
scaler = MinMaxScaler()
merged_data['volume_scaled'] = scaler.fit_transform(merged_data[['volume (ml)']])

# Perform Standardization on the '% of eTIV' feature
standard_scaler = StandardScaler()
merged_data['eTIV_standardized'] = standard_scaler.fit_transform(merged_data[['% of eTIV']])

## Normalization ##
# Perform normalization on the 'volume (ml)' feature using L2 norm
normalizer = Normalizer(norm='l2')
merged_data['volume_normalized'] = normalizer.transform(merged_data[['volume (ml)']])

## Binning ##
# Perform binning on the 'volume (ml)' feature
bin_encoder = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
merged_data['volume_binned'] = bin_encoder.fit_transform(merged_data[['volume (ml)']])

# Print the updated DataFrame with scaled, normalized, and binned features
print(merged_data[['volume (ml)', 'volume_scaled', 'eTIV_standardized', 'volume_normalized', 'volume_binned']])



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

# Assuming you have the true labels (y_test) and predicted labels (y_pred)
# Calculate the confusion matrix
#cm = confusion_matrix(y_test, y_pred)

# Get the class labels from the unique values in y_test and y_pred
#classes = sorted(set(y_test) | set(y_pred))

# Create a DataFrame from the confusion matrix
#cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Plot the confusion matrix using a heatmap
#plt.figure(figsize=(8, 6))
#sns.heatmap(cm_df, annot=True, cmap='Blues')
#plt.title('Confusion Matrix')
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.show()


import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Assuming you have the true labels (y_test) and predicted probabilities (y_pred_prob) for all classes
# Binarize the labels
y_test_bin = label_binarize(y_test, classes=[3, 4, 1, 2, 0])  # Replace class_labels with your actual class labels

# Get the class labels from the label encoder
class_labels = label_encoder.inverse_transform([3, 4, 1, 2, 0])  # Replace label_encoder with your actual label encoder

# Calculate the false positive rate (FPR), true positive rate (TPR), and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(class_labels)  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves for each class
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])  # Adjust the number of colors as per your classes

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'.format(class_labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


