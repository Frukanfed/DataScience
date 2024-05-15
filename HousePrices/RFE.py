import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Veri setlerini yükleyelim
train_df = pd.read_csv("./data/train_normalized_encoded.csv")

# Hedef değişkeni belirleyelim
target = "SalePrice"
X = train_df.drop(columns=[target])
y = train_df[target]

# Model oluştur
model = LinearRegression()

# RFE ile özellik seçimi
n_features_to_select = 20  # Seçilecek özellik sayısı
rfe = RFE(model, n_features_to_select=n_features_to_select)
fit = rfe.fit(X, y)

# Seçilen özellikleri belirleyelim
selected_features = X.columns[fit.support_]

# Seçilen özelliklere göre yeni veri setini oluşturalım
train_selected = train_df[selected_features.tolist() + [target]]

train_data, test_data = train_test_split(train_selected, test_size=0.2, random_state=42)

# Bölünmüş veri setlerini kaydedelim
train_data.to_csv("./updated_data/rfe_train_data.csv", index=False)
test_data.to_csv("./updated_data/rfe_test_data.csv", index=False)
