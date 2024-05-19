import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Veriyi yükle
data = pd.read_csv("./data.csv")

# String sütunları belirle
string_columns = data.select_dtypes(include=["object"]).columns

# String sütunları numaralandır
label_encoders = {}
for col in string_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Veri setini normalize et
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Yeni verileri kaydet
normalized_data.to_csv("./normalized_data.csv", index=False)
