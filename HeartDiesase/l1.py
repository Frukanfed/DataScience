import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# Normalize edilmiş veriyi oku
normalized_data = pd.read_csv("./data/normalized_heart_statlog.csv")

# Hedef sütunu ayır
target_column = "target"
X = normalized_data.drop(columns=[target_column])
y = normalized_data[target_column]

# Lasso (L1 Regularization) ile özellik seçimi yap
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
model = SelectFromModel(lasso, prefit=True)
X_lasso_selected = model.transform(X)

selected_features_lasso = X.columns[model.get_support()]
print("Lasso ile seçilen özellikler:", selected_features_lasso)

# Seçilen özelliklerle yeni veri seti oluşturma
selected_data = pd.concat([X[selected_features_lasso], y], axis=1)

# Veriyi %80 eğitim ve %20 test olarak ayır
train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42)

# Verileri kaydet
train_data.to_csv("./updated_data_training.csv", index=False)
test_data.to_csv("./updated_data/lasso_testing.csv", index=False)
