import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Perform one-hot encoding on categorical variables
categorical_columns = ['structure', 'type', 'hemisphere', 'diagnosis', 'sex', 'MRI', 'EEG']
encoded_data = pd.get_dummies(merged_data, columns=categorical_columns)

# Apply label encoding on 'Volume (ml)' and '% of eTIV'
label_encoder = LabelEncoder()
encoded_data['volume (ml)'] = label_encoder.fit_transform(merged_data['volume (ml)'])
encoded_data['% of eTIV'] = label_encoder.fit_transform(merged_data['% of eTIV'])

# Print the encoded data for verification
print(encoded_data)
