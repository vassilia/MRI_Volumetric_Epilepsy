import pandas as pd

# Assuming your data is stored in a DataFrame called 'data'
# with columns: 'structure', 'type', 'hemi', 'volume ml', '% of eTIV'

# Replace 'path_to_excel_file.xlsx' with the actual path to your Excel file
excel_file = 'mridata.xlsx'

# Read each sheet into a separate DataFrame
data_frames = pd.read_excel(excel_file, sheet_name=None)

# Calculate mean, median, and standard deviation of volume and % eTIV for each structure
structure_stats = {}
for sheet_name, df in data_frames.items():
    structure_stats[sheet_name] = df.groupby('structure')[['volume ml', '% of eTIV']].agg(['mean', 'median', 'std'])

# Determine frequency distribution of types and hemispheres within each structure
type_distribution = {}
hemisphere_distribution = {}
for sheet_name, df in data_frames.items():
    type_distribution[sheet_name] = df.groupby('structure')['type'].value_counts()
    hemisphere_distribution[sheet_name] = df.groupby('structure')['hemi'].value_counts()

# Print the calculated statistics and frequency distributions
print("Descriptive Statistics:")
for sheet_name, stats in structure_stats.items():
    print("Sheet Name:", sheet_name)
    print(stats)
    print("\n")

print("Type Distribution:")
for sheet_name, distribution in type_distribution.items():
    print("Sheet Name:", sheet_name)
    print(distribution)
    print("\n")

print("Hemisphere Distribution:")
for sheet_name, distribution in hemisphere_distribution.items():
    print("Sheet Name:", sheet_name)
    print(distribution)
    print("\n")
