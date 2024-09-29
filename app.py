import pandas as pd
import numpy as np
import seaborn as sns
import plotly_express as px
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("C:/Users/rache/OneDrive/Desktop/bahan PI/kodingan musik spotify/tracks.csv")

df.columns

df.describe()

df.info()

df.isnull().sum()

df = df.dropna(how='any', axis=0)
df

df[df.duplicated()].sum()

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month_name()

df

number_of_tracks_by_year = df.groupby(df['release_year'])['name'].count().reset_index()
number_of_tracks_by_year['Tracks released'] = number_of_tracks_by_year['name']
# Membuat diagram garis:
fig = px.line(number_of_tracks_by_year, x='release_year', y='Tracks released')
fig.update_layout(title="Lagu yang dirilis sepanjang tahun",title_x=0.5,
                  xaxis_title="Tahun rilis", yaxis_title="Jumlah lagu")

fig.show();

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
number_of_tracks_by_month = df.groupby(df['release_month'])['name'].count().reset_index()
# Membuat diagram batang:
plt.figure(figsize=(16, 5))
sns.barplot(x='release_month', y='name', data=number_of_tracks_by_month, order=month_order)
plt.xlabel('Bulan rilis')
plt.ylabel('Jumlah Lagu')
plt.title('Lagu yang dirilis berdasarkan bulan')
plt.xticks(rotation=45)
plt.show()

# Membuat diagram scatter plot:
plt.figure(figsize=(16,5))
sns.scatterplot(x = 'duration_ms', y = 'popularity', data = df)
plt.xlabel('Durasi')
plt.ylabel('Popularitas')
plt.title('Relasi antara durasi dan popularitas');

number_of_tracks = df.groupby(df['danceability'])['name'].count().reset_index()
number_of_tracks['popularity'] = number_of_tracks['name']
# Membuat diagram garis:
fig = px.line(number_of_tracks, x='danceability', y='popularity')
fig.update_layout(title="perbandingan popularitas dan danceability",title_x=0.5,
                  xaxis_title="danceability", yaxis_title="popularitas")

fig.show();

df=df.drop(columns='release_date')

month = {'January': 1,'February': 2 ,'March':3, "April":4, 'May':5, "June":6, 'July': 7, 'August':8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}


df['release_month'] = df['release_month'].fillna('Unknown')

df['release_month'] = df['release_month'].map(month)
print(df)

df

df_quantitative = df
cols_to_drop = []
for column in df:
    if df[column].dtype == 'object':
        cols_to_drop.append(column)
df_quantitative = df.drop(columns=cols_to_drop)

df_quan_2016_unnormalized = df_quantitative[df_quantitative['release_year']>=2016]
print(f"Working dataset shape: {df_quan_2016_unnormalized.shape}")

df_quan_2016_nm=(df_quan_2016_unnormalized-df_quan_2016_unnormalized.min())/(df_quan_2016_unnormalized.max()-df_quan_2016_unnormalized.min())

df_quan_2016_nm=df_quan_2016_nm.drop(columns='release_year')

np.random.seed(1)

df_train_full = df_quan_2016_nm.sample(frac=0.8,random_state=1)
df_test = df_quan_2016_nm.drop(df_train_full.index)

df_validation = df_train_full.sample(frac=0.2,random_state=2)
df_train = df_train_full.drop(df_validation.index)

predict = "popularity"
X_train = df_train.drop(columns=[predict])
X_validation = df_validation.drop(columns=[predict])
X_test = df_test.drop(columns=[predict])

Y_train = df_train[[predict]].values.ravel()
Y_validation = df_validation[[predict]].values.ravel()
Y_test = df_test[[predict]].values.ravel()

#menghitung mean squared error
def calculate_error(Y_pred, Y_actual):
    error = 0
    for i in range(len(Y_pred)):
        error += abs(Y_pred[i] - Y_actual[i])**2
    return error / len(Y_pred)

#menghitung mean absolute error
def calculate_mae(Y_pred, Y_actual):
    error = 0
    for i in range(len(Y_pred)):
        error += abs(Y_pred[i] - Y_actual[i])
    return error / len(Y_pred)

k_errors = [np.inf]
for k in range(1,50):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, Y_train)
    Y_val_pred = model.predict(X_validation)
    k_errors.append(calculate_error(Y_val_pred, Y_validation))

k_values = list(range(1, 51))
df = pd.DataFrame({'k': k_values, 'error': k_errors})

fig = px.bar(df, x='k', y='error', title="Nilai kesalahan untuk nilai k yang berbeda pada regressor KNN")
fig.update_layout(xaxis_title="Nilai k", yaxis_title="Error")
fig.show()

k=7
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# menampilkan MSE and MAE
mse = calculate_error(Y_pred, Y_test)
mae = calculate_mae(Y_pred, Y_test)

print(f"Hasil tes mencatat MSE sebesar: {mse}")
print(f"Hasil tes mencatat MAE sebesar: {mae}")

# prompt: menguji coba kalau program terbukti berjalan dengan baik

# Memprediksi popularitas untuk satu lagu di set pengujian
index_to_predict = 5  # Ubah ini ke indeks lagu yang ingin Anda prediksi
single_track_data = X_test.iloc[[index_to_predict]]
predicted_popularity = model.predict(single_track_data)

print(f"Popularitas yang diprediksi: {predicted_popularity[0]}")
print(f"Popularitas sebenarnya: {Y_test[index_to_predict]}")

# Menampilkan baris ke-2 dari DataFrame (misalnya, df atau X_test)
row_2 = df.iloc[1]  # Mengambil baris ke-2 dari DataFrame
print(row_2)

print(df_test.head(10))  # Adjust the number to see more rows

# Access the row at index 12
row_at_index_5 = df_test.iloc[5]

# Print the information for that song
print(row_at_index_5)

# Menampilkan 2 baris terakhir dari X_test
last_row_index = X_test.shape[0] - 2  # Mendapatkan indeks baris terakhir
row_last = X_test.iloc[last_row_index]
print(row_last)

# Memprediksi popularitas untuk dua baris terakhir dari X_test
single_track_data_last_two = X_test.iloc[-2:]  # Mengambil dua baris terakhir

# Memprediksi popularitas untuk dua baris terakhir
predicted_popularity_last_two = model.predict(single_track_data_last_two)

# Menampilkan hasil prediksi dan nilai sebenarnya
for i in range(1):
    print(f"Popularitas yang diprediksi untuk baris {X_test.index[-2+i]}: {predicted_popularity_last_two[i]}")
    print(f"Popularitas sebenarnya: {Y_test[-2+i]}")

# prompt: Memprediksi popularitas untuk baris kedua terakhir dari X_test

# Mengambil baris kedua terakhir dari X_test
second_last_row_index = X_test.shape[0] - 2
single_track_data_second_last = X_test.iloc[[second_last_row_index]]

# Memprediksi popularitas untuk baris kedua terakhir
predicted_popularity_second_last = model.predict(single_track_data_second_last)

# Menampilkan hasil prediksi dan nilai sebenarnya
print(f"Popularitas yang diprediksi untuk baris {X_test.index[second_last_row_index]}: {predicted_popularity_second_last[0]}")
print(f"Popularitas sebenarnya: {Y_test[second_last_row_index]}")