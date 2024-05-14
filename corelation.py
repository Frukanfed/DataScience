import pandas as pd

# Veri setlerini yükleyelim
train_df = pd.read_csv("./data/train_normalized_encoded.csv")

# Hedef değişkeni belirleyelim (örneğin 'SalePrice')
target = "SalePrice"

# Korelasyon matrisi hesaplayalım
correlation_matrix = train_df.corr()

# Hedef değişken ile diğer sütunlar arasındaki korelasyonları seçelim
correlation_with_target = correlation_matrix[target].drop(target)

# 81 faktörden 0.5 belirleyince 18'e, 0.2 belirleyince 38'e, 0.3 belirleyince 27'ye düşüyor
threshold = 0.3

# Eşik değerine göre yüksek korelasyona sahip özellikleri seçelim
selected_features = correlation_with_target[
    abs(correlation_with_target) > threshold
].index.tolist()

# Seçilen özellikleri ve hedef değişkeni içeren yeni veri seti oluşturalım
selected_features.append(target)
train_selected = train_df[selected_features]

# Seçilen özelliklere göre veri setini kaydedelim
train_selected.to_csv("./updated_data/corelation_train_data.csv", index=False)

train_selected.head()
