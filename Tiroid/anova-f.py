import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

# Veri setini yükleyelim
train_df = pd.read_csv("./normalized_data.csv")

# Hedef değişkeni belirleyelim
target = "Recurred"
X = train_df.drop(columns=[target])
y = train_df[target]

# ANOVA F-Test ile özellik seçimi
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X, y)

# Seçilen özellikleri belirleyelim
selected_features = X.columns[selector.get_support()].tolist()

# Seçilen özelliklere göre yeni veri setini oluşturalım
train_selected = train_df[selected_features + [target]]

# Veriyi %80-%20 oranında bölelim
train_data, test_data = train_test_split(train_selected, test_size=0.2, random_state=42)

# Bölünmüş veri setlerini kaydedelim
train_data.to_csv("./feature_selected_data/anova_train.csv", index=False)
test_data.to_csv("./feature_selected_data/anova_test.csv", index=False)
