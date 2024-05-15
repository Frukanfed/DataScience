import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Veri setlerini yükleyelim
train_df = pd.read_csv("./data/train.csv")

# 'Id' sütununu koruyalım
train_id = train_df["Id"]
train_SalePrice = train_df["SalePrice"]

# 'Id' sütununu veri setlerinden çıkaralım
train_df = train_df.drop(columns=["Id", "SalePrice"])

# Sayısal sütunları tespit edelim (object olmayanlar)
numeric_columns_train = train_df.select_dtypes(include=["int64", "float64"]).columns

# Min-Max Scaler oluştur
scaler = MinMaxScaler()

# Sayısal sütunları normalleştirelim
train_df[numeric_columns_train] = scaler.fit_transform(train_df[numeric_columns_train])

# Tüm string sütunları tespit edelim
string_columns_train = train_df.select_dtypes(include=["object"]).columns

# Label Encoder oluştur
le = LabelEncoder()

# String sütunlara Label Encoding uygulayalım
for col in string_columns_train:
    train_df[col] = le.fit_transform(train_df[col].astype(str))

# NaN değerlerini -1 ile dolduralım
train_df.fillna(-1, inplace=True)

# 'Id' sütununu geri ekleyelim
train_df["Id"] = train_id
train_df["SalePrice"] = train_SalePrice

# Sütunları tekrar sıralayalım
train_df = train_df[["Id"] + [col for col in train_df.columns if col != "Id"]]

# Dönüştürülmüş veri setlerini CSV dosyalarına kaydedelim
train_df.to_csv("./data/train_normalized_encoded.csv", index=False)
