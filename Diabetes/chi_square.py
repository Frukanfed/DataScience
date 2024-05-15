import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Load the normalized and encoded dataset
df = pd.read_csv("./data/train_normalized_encoded.csv")

# Define the features and the target
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Apply SelectKBest with chi-squared (chi2)
# For example, we can select the top 6 features
selector = SelectKBest(score_func=chi2, k=6)
X_new = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]

# Create a new dataframe with the selected features and the target
df_selected = pd.DataFrame(X_new, columns=selected_features)
df_selected["diabetes"] = y.values

# Display the selected features
print("Selected features:", selected_features.tolist())
print(df_selected.head())

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Save the datasets to CSV files
train_df.to_csv("./updated_data/chi_square_train.csv", index=False)
test_df.to_csv("./updated_data/chi_square_test.csv", index=False)
