import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Veriyi oku
data = pd.read_csv("./data.csv")

# NaN verileri kontrol et
nan_check = data.isna().sum()

# MinMaxScaler ile normalizasyon uygula
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Sonuçları yazdır
print("NaN Veriler:")
print(nan_check)

# NaN veri olmadığı için veriyi kaydet
normalized_data.to_csv("./newdata.csv", index=False)
