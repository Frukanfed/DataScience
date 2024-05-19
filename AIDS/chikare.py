import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Veriyi yükle
df = pd.read_csv("./newdata.csv")

# Hedef ve özellikleri tanımla
X = df.drop(columns=["infected"])
y = df["infected"]

# Modeli hazırla(chi2) ve seçilecek özellik sayısını seç
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)

# Seçilen özellikleri selector.get_support() ile kaydet
selected_features = X.columns[selector.get_support()]

# Seçilen özelliklerle yeni veri seti oluşturma
df_selected = pd.DataFrame(X_new, columns=selected_features)
df_selected["infected"] = y.values

# Veriyi %80 eğitim ve %20 test olarak ayır
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Verileri kaydet
train_df.to_csv("./veriler/chikare.csv", index=False)
test_df.to_csv("./veriler/chikaretest.csv", index=False)
