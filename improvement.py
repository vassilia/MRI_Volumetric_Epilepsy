import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, Normalizer

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
