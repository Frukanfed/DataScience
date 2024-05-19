import pandas as pd
from sklearn.model_selection import train_test_split

# Yeni CSV dosyasını yükleyelim
df = pd.read_csv("./normalized_data.csv")

# Korelasyon matrisini hesaplayalım
correlation_matrix = df.corr()

# 'Recurred' sütunu ile olan korelasyonları alalım
target_correlation = correlation_matrix["Recurred"].sort_values(ascending=False)

# Toplan veri 16. 0.5 - 4 özellik, 0.3 - 9 özellik, 0.1 - 14 özellik
threshold = 0.3  # Bu değeri istediğinize göre ayarlayabilirsiniz
selected_features = target_correlation[abs(target_correlation) > threshold].index

# Seçilen özellikleri kontrol edelim
print("Seçilen özellikler:")
print(selected_features)

# Seçilen özelliklere sahip yeni veri setini oluşturalım
df_selected = df[selected_features]

# Veriyi eğitim ve test setlerine ayıralım
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Eğitim ve test verilerini kaydedelim
train_file_path = "./feature_selected_data/correlation_train.csv"
test_file_path = "./feature_selected_data/correlation_test.csv"

train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
