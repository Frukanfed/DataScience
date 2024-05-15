import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Veriyi yükle
df = pd.read_csv("./data/diabetes_prediction.csv")

# gender sütununu integer değerlere çevir: Female -> 0, Male -> 1
df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

# smoking_history sütununu integer değerlere çevir
smoking_mapping = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5,
}
df["smoking_history"] = df["smoking_history"].map(smoking_mapping)

# NaN değerleri kontrol et
print("NaN values in dataframe before dropping:")
print(df.isna().sum())

# NaN değerleri sil
df = df.dropna()

# NaN değerleri tekrar kontrol et
print("NaN values in dataframe after dropping:")
print(df.isna().sum())


# Ekstrem değerleri ayıklamak için IQR yöntemini kullan
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# 'diabetes' sütunu hariç sayısal sütunları seç
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.drop(
    "diabetes"
)
for col in numeric_columns:
    df = remove_outliers_iqr(df, col)

# Normalleştirilecek sütunları seç, 'diabetes' hedef değişkeni hariç
columns_to_normalize = df.columns[df.columns != "diabetes"]

# MinMaxScaler'ı başlat
scaler = MinMaxScaler()

# Seçilen sütunları normalize et
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Temizlenmiş ve normalize edilmiş veri çerçevesini yeni bir CSV dosyasına kaydet
df.to_csv("./data/train_normalized_encoded.csv", index=False)
