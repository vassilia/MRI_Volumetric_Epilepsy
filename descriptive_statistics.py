import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data from Excel file
df = pd.read_excel('new_mri_data.xlsx')

# Define a function to replace commas with decimal points and convert to numeric type
def convert_to_numeric(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return value

# Apply the conversion function to the 'volume ml' and '% of eTIV' columns
df['volume ml'] = df['volume ml'].apply(convert_to_numeric)
df['% of eTIV'] = df['% of eTIV'].apply(convert_to_numeric)

# Replace NaN values with a default value (e.g., 0)
df.fillna(0, inplace=True)

# Check the data types and summary statistics
print(df.dtypes)
print(df.describe())
# Replace 'path_to_excel_file.xlsx' with the actual path to your Excel file
excel_file = 'mridata.xlsx'

# Read each sheet into a separate DataFrame
data_frames = pd.read_excel(excel_file, sheet_name=None)

# Combine all sheets into a single DataFrame
combined_data = pd.concat(data_frames.values(), ignore_index=True)

# Fill missing values with a default value or impute them using appropriate techniques
combined_data['volume ml'].fillna(0, inplace=True)  # Fill missing volume values with 0

# Calculate volume asymmetry percent for each structure
combined_data['Volume Asymmetry %'] = ((combined_data[combined_data['hemi'] == 'right']['volume ml'] - combined_data[combined_data['hemi'] == 'left']['volume ml']) / ((combined_data[combined_data['hemi'] == 'right']['volume ml'] + combined_data[combined_data['hemi'] == 'left']['volume ml']) / 2)) * 100

# Analyze the distribution of volume asymmetry across different structures
structure_asymmetry = combined_data.groupby('structure')['Volume Asymmetry %'].mean()

# Visualize the distribution of volume asymmetry using a histogram
plt.hist(combined_data['Volume Asymmetry %'], bins=10)
plt.title("Distribution of Volume Asymmetry")
plt.xlabel("Volume Asymmetry (%)")
plt.ylabel("Frequency")
plt.show()

# Calculate mean and 95% CI volume / TIV-adjusted volume
combined_data['Volume TIV-Adjusted %'] = (combined_data['volume ml'] / combined_data['% of eTIV']) * 100
mean_volume_adjusted = combined_data.groupby('structure')['Volume TIV-Adjusted %'].mean()
ci_volume_adjusted = combined_data.groupby('structure')['Volume TIV-Adjusted %'].agg(lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5))

# Print the results
print("Volume Asymmetry:")
print(structure_asymmetry)
print("\nMean & 95% CI Volume TIV-Adjusted (%):")
print(mean_volume_adjusted)
print(ci_volume_adjusted)

# Calculate mean, median, and standard deviation of volume for each structure
structure_stats = combined_data.groupby('structure')['volume ml'].agg(['mean', 'median', 'std'])

# Visualize the calculated statistics
structure_stats.plot(kind='bar', y=['mean', 'median', 'std'], rot=0)
plt.title("Volume Statistics by Structure")
plt.xlabel("Structure")
plt.ylabel("Volume (ml)")
plt.legend(["Mean", "Median", "Std"])
plt.show()
