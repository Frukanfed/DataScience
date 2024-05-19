import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Normalize edilmiş veriyi oku
normalized_data = pd.read_csv("./newdata.csv")

# Hedef sütunu ayır
X = normalized_data.drop(columns=["infected"])
y = normalized_data["infected"]

# Korelasyon matrisini hesapla ve görselleştir
correlation_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# 22 özellik var. 0.5 seçilince 20 özellik, 0.3 seçilince 18 özellik, 0.1 seçince 12 özellik 0.2 seçince 14 özellik
correlation_threshold = 0.2
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
train_data.to_csv("./veriler/korelasyon.csv", index=False)
test_data.to_csv("./veriler/korelasyontest.csv", index=False)
