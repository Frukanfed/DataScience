import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load the normalized and encoded dataset
df = pd.read_csv("./data/train_normalized_encoded.csv")

# Define the features and the target
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Initialize the model
model = LogisticRegression()

rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[fit.support_]

# Create a new dataframe with the selected features and the target
df_selected = pd.concat([X[selected_features], y], axis=1)

# Display the selected features
print("Selected features:", selected_features.tolist())
print(df_selected.head())

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Save the datasets to CSV files
train_df.to_csv("./updated_data/rfe_train.csv", index=False)
test_df.to_csv("./updated_data/rfe_test.csv", index=False)
