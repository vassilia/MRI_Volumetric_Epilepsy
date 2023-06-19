import pandas as pd

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



