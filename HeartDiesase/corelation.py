import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Normalize edilmiş veriyi oku
normalized_data = pd.read_csv("./data/normalized_heart_statlog.csv")

# Hedef sütunu ayır
target_column = "target"
X = normalized_data.drop(columns=[target_column])
y = normalized_data[target_column]

# Korelasyon matrisini hesapla ve görselleştir
correlation_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# Korelasyon eşiğine göre özellik seçimi (örneğin, 0.5 üzerinde)
correlation_threshold = 0.5
selected_features = set(X.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            if colname in selected_features:
                selected_features.remove(colname)

print("Korelasyon ile seçilen özellikler:", selected_features)

# Seçilen özelliklerle yeni veri seti oluşturma
selected_data = pd.concat([X[list(selected_features)], y], axis=1)

# Veriyi %80 eğitim ve %20 test olarak ayır
train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42)

# Verileri kaydet
train_data.to_csv("./updated_data/correlation_training.csv", index=False)
test_data.to_csv("./updated_data/correlation_testing.csv", index=False)

print("Eğitim verisi kaydedildi: correlation_training.csv")
print("Test verisi kaydedildi: correlation_testing.csv")
