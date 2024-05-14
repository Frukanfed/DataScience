import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Veri setlerini yükleyelim
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# 'Id' sütununu koruyalım
train_id = train_df["Id"]
train_SalePrice = train_df["SalePrice"]
test_id = test_df["Id"]

# 'Id' sütununu veri setlerinden çıkaralım
train_df = train_df.drop(columns=["Id", "SalePrice"])
test_df = test_df.drop(columns=["Id"])

# Sayısal sütunları tespit edelim (object olmayanlar)
numeric_columns_train = train_df.select_dtypes(include=["int64", "float64"]).columns
numeric_columns_test = test_df.select_dtypes(include=["int64", "float64"]).columns

# Min-Max Scaler oluştur
scaler = MinMaxScaler()

# Sayısal sütunları normalleştirelim
train_df[numeric_columns_train] = scaler.fit_transform(train_df[numeric_columns_train])
test_df[numeric_columns_test] = scaler.fit_transform(test_df[numeric_columns_test])

# Tüm string sütunları tespit edelim
string_columns_train = train_df.select_dtypes(include=["object"]).columns
string_columns_test = test_df.select_dtypes(include=["object"]).columns

# Label Encoder oluştur
le = LabelEncoder()

# String sütunlara Label Encoding uygulayalım
for col in string_columns_train:
    train_df[col] = le.fit_transform(train_df[col].astype(str))

for col in string_columns_test:
    test_df[col] = le.fit_transform(test_df[col].astype(str))

# NaN değerlerini -1 ile dolduralım
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)

# 'Id' sütununu geri ekleyelim
train_df["Id"] = train_id
train_df["SalePrice"] = train_SalePrice
test_df["Id"] = test_id

# Sütunları tekrar sıralayalım
train_df = train_df[["Id"] + [col for col in train_df.columns if col != "Id"]]
test_df = test_df[["Id"] + [col for col in test_df.columns if col != "Id"]]

# Dönüştürülmüş veri setlerini CSV dosyalarına kaydedelim
train_df.to_csv("./data/train_normalized_encoded.csv", index=False)
test_df.to_csv("./data/test_normalized_encoded.csv", index=False)

train_df.head(), test_df.head()
