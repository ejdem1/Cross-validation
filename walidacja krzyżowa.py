import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Wczytaj dane
data = pd.read_excel('US30_H4_201601260000_202410251600.xlsx')

# Upewnij się, że nazwy kolumn są zapisane wielkimi literami
data.columns = data.columns.str.upper()

# Oblicz procentowe zmiany ceny zamknięcia
data['CLOSE_PCT_CHANGE'] = data['CLOSE'].pct_change()

# Oblicz procentowe zmiany TICKVOL
data['TICKVOL_PCT_CHANGE'] = data['TICKVOL'].pct_change()

# Utwórz opóźnione wersje TICKVOL_PCT_CHANGE
data['TICKVOL_PCT_CHANGE_LAG1'] = data['TICKVOL_PCT_CHANGE'].shift(1)
data['TICKVOL_PCT_CHANGE_LAG2'] = data['TICKVOL_PCT_CHANGE'].shift(2)
data['TICKVOL_PCT_CHANGE_LAG3'] = data['TICKVOL_PCT_CHANGE'].shift(3)

# Oblicz wskaźniki techniczne
data['SMA_10'] = data['CLOSE'].rolling(window=10).mean()
data['EMA_10'] = data['CLOSE'].ewm(span=10, adjust=False).mean()
delta = data['CLOSE'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))
ema_12 = data['CLOSE'].ewm(span=12, adjust=False).mean()
ema_26 = data['CLOSE'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['MACD_DIFF'] = data['MACD'] - data['MACD'].ewm(span=9, adjust=False).mean()
sma_20 = data['CLOSE'].rolling(window=20).mean()
std_20 = data['CLOSE'].rolling(window=20).std()
data['BOLLINGER_HIGH'] = sma_20 + (std_20 * 2)
data['BOLLINGER_LOW'] = sma_20 - (std_20 * 2)

# Nowe wskaźniki techniczne
data['TR'] = data[['HIGH', 'LOW', 'CLOSE']].max(axis=1) - data[['HIGH', 'LOW', 'CLOSE']].min(axis=1)
data['ATR'] = data['TR'].rolling(window=14).mean()

data['L14'] = data['LOW'].rolling(window=14).min()
data['H14'] = data['HIGH'].rolling(window=14).max()
data['%K'] = 100 * ((data['CLOSE'] - data['L14']) / (data['H14'] - data['L14']))
data['%D'] = data['%K'].rolling(window=3).mean()

data['TP'] = (data['HIGH'] + data['LOW'] + data['CLOSE']) / 3
data['CCI'] = (data['TP'] - data['TP'].rolling(window=20).mean()) / (0.015 * data['TP'].rolling(window=20).std())

data['MOMENTUM'] = data['CLOSE'] - data['CLOSE'].shift(4)
data['ROC'] = ((data['CLOSE'] - data['CLOSE'].shift(12)) / data['CLOSE'].shift(12)) * 100
data['WILLIAMS %R'] = (data['H14'] - data['CLOSE']) / (data['H14'] - data['L14']) * -100

data['DM+'] = data['HIGH'].diff()
data['DM-'] = data['LOW'].diff()
data['DM+'] = data['DM+'][data['DM+'] > data['DM-']]
data['DM-'] = data['DM-'][data['DM-'] > data['DM+']]
data['DM+'] = data['DM+'].fillna(0)
data['DM-'] = data['DM-'].fillna(0)
data['TR14'] = data['TR'].rolling(window=14).sum()
data['DM+14'] = data['DM+'].rolling(window=14).sum()
data['DM-14'] = data['DM-'].rolling(window=14).sum()
data['DI+14'] = (data['DM+14'] / data['TR14']) * 100
data['DI-14'] = (data['DM-14'] / data['TR14']) * 100
data['DMI'] = abs(data['DI+14'] - data['DI-14']) / (data['DI+14'] + data['DI-14']) * 100

# Usuń wiersze z brakującymi wartościami
data.dropna(inplace=True)

# Utwórz zmienną docelową
data['CLOSE_DIRECTION'] = (data['CLOSE_PCT_CHANGE'] > 0).astype(int)

# Zdefiniuj cechy i zmienną docelową
X = data[['TICKVOL_PCT_CHANGE', 'TICKVOL_PCT_CHANGE_LAG1', 'TICKVOL_PCT_CHANGE_LAG2', 'TICKVOL_PCT_CHANGE_LAG3',
          'SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_DIFF', 'BOLLINGER_HIGH', 'BOLLINGER_LOW',
          'ATR', '%K', '%D', 'CCI', 'MOMENTUM', 'ROC', 'WILLIAMS %R', 'DMI']]
y = data['CLOSE_DIRECTION']

# Zastosuj SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standaryzuj cechy
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Stwórz model Random Forest z najlepszymi parametrami
rf_model_best = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=True,
    random_state=42
)

# Przeprowadź walidację krzyżową
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model_best, X_resampled_scaled, y_resampled, cv=cv, scoring='accuracy')

# Wyświetl wyniki walidacji krzyżowej
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Dopasuj model do całego zestawu danych i sprawdź metryki
rf_model_best.fit(X_resampled_scaled, y_resampled)
y_pred = rf_model_best.predict(X_resampled_scaled)

accuracy = accuracy_score(y_resampled, y_pred)
classification_rep = classification_report(y_resampled, y_pred)
conf_matrix = confusion_matrix(y_resampled, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(conf_matrix)
