import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

# Normalize edilmiş veriyi oku
normalized_data = pd.read_csv("./data/normalized_heart_statlog.csv")

# Hedef sütunu ayır
target_column = "target"
X = normalized_data.drop(columns=[target_column])
y = normalized_data[target_column]

# Chi-Squared testi ile özellik seçimi yap
chi2_selector = SelectKBest(chi2, k=8)  # En iyi 8 özelliği seç
X_chi2_selected = chi2_selector.fit_transform(X, y)

selected_features_chi2 = X.columns[chi2_selector.get_support()]
print("Chi-Squared ile seçilen özellikler:", selected_features_chi2)

# Seçilen özelliklerle yeni veri seti oluşturma
selected_data = pd.concat([X[selected_features_chi2], y], axis=1)

# Veriyi %80 eğitim ve %20 test olarak ayır
train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42)

# Verileri kaydet
train_data.to_csv("./updated_data/chi2_training.csv", index=False)
test_data.to_csv("./updated_data/chi2_testing.csv", index=False)
