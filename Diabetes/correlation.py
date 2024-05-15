import pandas as pd
from sklearn.model_selection import train_test_split

# Load the normalized and encoded dataset
df = pd.read_csv("./data/train_normalized_encoded.csv")

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Get the absolute value of the correlation with the target variable 'diabetes'
correlation_with_target = correlation_matrix["diabetes"].abs()

# 0.3 -> 0 özellik, 0.1 -> 4 özellik
threshold = 0.1
selected_features = correlation_with_target[correlation_with_target > threshold].index

# Create a new dataframe with the selected features
df_selected = df[selected_features]

# Display the selected features
print("Selected features:", selected_features.tolist())
print(df_selected.head())

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Save the datasets to CSV files
train_df.to_csv("./updated_data/correlation_train.csv", index=False)
test_df.to_csv("./updated_data/correlation_test.csv", index=False)
