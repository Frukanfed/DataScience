import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Veriyi yükle
df = pd.read_csv("./newdata.csv")

# Hedef ve özellikleri tanımla
X = df.drop(columns=["infected"])
y = df["infected"]

# Modeli hazırla
model = LogisticRegression()

rfe = RFE(model, n_features_to_select=15)
fit = rfe.fit(X, y)

# Seçilen özellikleri fit.support_ ile kaydet
selected_features = X.columns[fit.support_]

# Seçilen özelliklerle yeni veri seti oluşturma
df_selected = pd.concat([X[selected_features], y], axis=1)

# Veriyi %80 eğitim ve %20 test olarak ayır
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Verileri kaydet
train_df.to_csv("./veriler/rfe.csv", index=False)
test_df.to_csv("./veriler/rfetest.csv", index=False)
