# Laporan Proyek Machine Learning Terapan | Husnul Khatimah

# link file readme.md di github : https://github.com/husnlkhatmh/machine-learning-terapan1
 
## Domain Proyek : Kesehatan (🧬Klasifikasi Jenis Anemia) 

## Latar Belakang
Anemia masih menjadi masalah kesehatan yang cukup besar di dunia, terutama pada anak-anak, remaja, dan wanita usia subur. Kondisi ini terjadi ketika kadar hemoglobin atau jumlah sel darah merah dalam tubuh menurun, sehingga kemampuan darah membawa oksigen ke jaringan tubuh tidak optimal. Berdasarkan laporan WHO tahun 2021, sekitar 42% anak balita dan 40% ibu hamil di seluruh dunia mengalami anemia, sehingga masalah ini tetap menjadi perhatian serius di bidang kesehatan (WHO, 2021).

Biasanya, anemia dideteksi lewat pemeriksaan Complete Blood Count (CBC) yang mengukur beberapa indikator penting seperti kadar hemoglobin (HGB), jumlah sel darah merah (RBC), ukuran rata-rata sel darah (MCV), dan konsentrasi hemoglobin dalam sel darah merah (MCHC). Walaupun pemeriksaan ini cukup akurat, proses membaca dan menganalisis data CBC secara manual sering kali memerlukan waktu lama dan tenaga ahli yang tidak selalu tersedia, terutama di fasilitas kesehatan dengan sumber daya terbatas.

Teknologi machine learning hadir sebagai solusi yang bisa mempercepat dan mempermudah proses klasifikasi jenis anemia berdasarkan data CBC. Dengan algoritma yang mampu mengenali pola dalam data, ML dapat membantu mengotomatisasi diagnosis sehingga hasilnya lebih cepat dan dapat diandalkan. Studi oleh Bazgir dan tim (2019) menunjukkan bahwa penggunaan machine learning cukup efektif dalam mengklasifikasi tipe-tipe anemia berdasarkan data medis (Bazgir et al., 2019). Selain itu, penelitian lain oleh Khanday dkk. (2020) mengungkapkan bahwa algoritma seperti Random Forest dan XGBoost memberikan hasil klasifikasi yang akurat dan dapat diandalkan pada data kesehatan yang kompleks (Khanday et al., 2020).

Dalam proyek ini, dikembangkan sistem klasifikasi anemia menggunakan machine learning dengan data CBC, yang diharapkan bisa membantu tenaga medis dalam mempercepat proses diagnosis.

## Business Understanding
### Problem Statements
-	Bagaimana cara mengembangkan model machine learning yang mampu mengklasifikasikan jenis anemia berdasarkan data hasil tes darah lengkap (CBC)?
-	Model algoritma machine learning apa yang memberikan hasil terbaik dalam klasifikasi jenis anemia pada dataset ini?
- Bagaimana melakukan evaluasi menyeluruh terhadap performa model agar dapat memastikan klasifikasi jenis anemia berjalan optimal?
-	Seberapa besar peningkatan akurasi model klasifikasi setelah dilakukan proses hyperparameter tuning?

### Goals
-	Membangun model klasifikasi machine learning yang dapat secara akurat mengidentifikasi jenis anemia menggunakan data CBC.
-	Membandingkan performa beberapa algoritma ML untuk klasifikasi jenis anemia.
-	Melakukan evaluasi menyeluruh terhadap performa model menggunakan berbagai metrik seperti akurasi, presisi, recall, F1-score, dan confusion matrix untuk memastikan model bekerja secara optimal dan seimbang dalam mengklasifikasikan jenis anemia.
-	Meningkatkan performa model melalui proses hyperparameter tuning.

### Solution Statement
-	Penggunaan beragam algoritma machne learning
Mencoba beberapa model machine learning seperti **Random Forest**, **XGBoost**, dan **SVM**, untuk melihat mana yang paling efektif dalam mengenali pola dari data CBC dan memprediksi jenis anemia secara akurat.
-	Meningkatkan Performa Model dengan Hyperparameter Tuning
Setelah model baseline dibuat, proses tuning dilakukan menggunakan **GridSearchCV** untuk mencari pengaturan parameter terbaik dari masing-masing model. Tujuannya adalah meningkatkan kinerja model agar hasil prediksi lebih baik.

## Data Understanding
Link dataset : https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification 

Dataset ini terdiri dari **1281 data (baris)** dan **15 kolom (fitur)**, yang berisi data hasil **Complete Blood Count (CBC)** atau hitung darah lengkap yang telah diberi label berdasarkan diagnosis jenis anemia. Data dikumpulkan dari berbagai hasil tes CBC dan setiap entri telah melalui proses diagnosis manual oleh tenaga medis.

| Fitur     | Deskripsi |
|-----------|-----------|
| **WBC**   | Jumlah sel darah putih, elemen utama dalam sistem kekebalan tubuh. |
| **LYMp**  | Persentase limfosit dalam darah dari total sel darah putih. |
| **NEUTp** | Persentase neutrofil dalam darah dari total sel darah putih. |
| **LYMn**  | Jumlah absolut limfosit dalam darah. |
| **NEUTn** | Jumlah absolut neutrofil dalam darah. |
| **RBC**   | Jumlah sel darah merah, bertugas membawa oksigen ke seluruh jaringan tubuh. |
| **HGB**   | Kadar hemoglobin dalam darah, berperan penting dalam pengangkutan oksigen ke seluruh tubuh. |
| **HCT**   | Hematokrit, persentase volume sel darah merah terhadap volume darah total. |
| **MCV**   | Rata-rata volume dari sel darah merah, digunakan untuk mengidentifikasi tipe anemia. |
| **MCH**   | Rata-rata jumlah hemoglobin dalam satu sel darah merah. |
| **MCHC**  | Konsentrasi rata-rata hemoglobin dalam sel darah merah. |
| **PLT**   | Jumlah trombosit, berfungsi dalam proses pembekuan darah. |
| **PDW**   | Ukuran variabilitas dari trombosit dalam darah. |
| **PCT**   | Kadar prokalsitonin, indikator potensial adanya infeksi bakteri atau risiko sepsis. |
| **Diagnosis** | Label klasifikasi yang menunjukkan jenis anemia berdasarkan hasil pemeriksaan darah lengkap. |

## Exploratory Data Analysis
**Informasi data :**

![image](https://github.com/user-attachments/assets/be027270-303a-4e73-be1e-8a6e68bc046d)

Seluruh fitur numerik dalam dataset bertipe **float64**, yang berarti setiap nilai pada kolom tersebut merupakan angka desimal hasil pengukuran dari tes darah lengkap. Sementara itu, kolom **Diagnosis** merupakan satu-satunya fitur bertipe **object** karena berisi kategori jenis anemia yang akan menjadi label atau target dalam proses klasifikasi. Dataset ini tidak memiliki nilai yang hilang (missing value). Tetapi dalam dataset ini terdapat data duplikat sebanyak 49, karena jumlah duplikat sedikit dibandingkan total data yang cukup besar, maka data duplikat saya hapus.

Informasi statistik pada masing masing kolom :

|index|WBC|LYMp|NEUTp|LYMn|NEUTn|RBC|HGB|HCT|MCV|MCH|MCHC|PLT|PDW|PCT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|1232\.0|
|mean|7\.846712662337664|25\.897658279220778|77\.59197808441557|1\.8843406818181823|5\.130724172077922|4\.714293831168831|12\.187589285714287|46\.26914951298701|85\.73409902597403|32\.25087662337663|31\.739732142857147|228\.92792207792203|14\.340931044650974|0\.2616226948051948|
|std|3\.5521801164120803|7\.064941470718204|150\.65159610836122|1\.357361856453615|2\.8956363648424968|2\.867519354898482|3\.863201252285587|106\.9459815942016|27\.663901528272948|113\.35665259387913|3\.3545115190991415|93\.08025833580771|3\.0538677967649415|0\.6987346777634921|
|min|0\.8|6\.2|0\.7|0\.2|0\.5|1\.36|-10\.0|2\.0|-79\.3|10\.9|11\.5|10\.0|8\.4|0\.01|
|25%|6\.0|25\.845|70\.775|1\.88076|5\.0|4\.19|10\.8|39\.2|81\.0|25\.5|30\.5|157\.0|13\.3|0\.17|
|50%|7\.4|25\.845|77\.511|1\.88076|5\.14094|4\.6|12\.2|46\.1526|86\.55|27\.7|32\.0|211\.0|14\.31251157|0\.26028|
|75%|8\.7|25\.845|77\.511|1\.88076|5\.14094|5\.1|13\.5|46\.1526|90\.2|29\.6|32\.9|290\.0|14\.8|0\.26028|
|max|45\.7|91\.4|5317\.0|41\.8|79\.0|90\.8|87\.1|3715\.0|990\.0|3117\.0|92\.8|660\.0|97\.0|13\.6|

**Analisi Distribusi Data**

![image](https://github.com/user-attachments/assets/fa5aab7d-9c32-4adf-8cc6-6e8cf659aa7e)

Grafik di atas menunjukkan distribusi jumlah kasus untuk setiap kategori diagnosis anemia. Dari grafik tersebut, terlihat bahwa kategori dengan jumlah kasus terbanyak adalah Healthy sebanyak 323 kasus, diikuti oleh Normocytic hypochromic anemia sebanyak 271 kasus dan Normocytic normochromic anemia sebanyak 255 kasus. Sementara itu, beberapa kategori memiliki jumlah kasus yang relatif sedikit, seperti Leukemia with thrombocytopenia (11 kasus), Macrocytic anemia (16 kasus), dan Leukemia (44 kasus).      

 ![image](https://github.com/user-attachments/assets/bb306779-5f48-466b-ac0f-fbb1d995b0f9)

Gambar di atas menunjukkan distribusi histogram dari fitur-fitur numerik dalam dataset darah lengkap (CBC) untuk klasifikasi anemia. Sebagian besar fitur seperti WBC, LYM, NEUT, RBC, dan PCT memiliki distribusi yang miring ke kanan, menandakan adanya dominasi nilai rendah dan keberadaan outlier. Beberapa fitur seperti PLT menunjukkan distribusi yang lebih seimbang. Pola distribusi ini menunjukkan bahwa sebagian besar data tidak terdistribusi normal.

![image](https://github.com/user-attachments/assets/e60fc31c-9712-4c3d-96eb-6e3bf8366423)

Saya memutuskan untuk tidak menghapus outlier karena data yang digunakan merupakan data medis untuk klasifikasi tipe anemia. Dalam kasus seperti ini, nilai-nilai ekstrem pada parameter darah bisa menjadi indikator penting dari tipe anemia tertentu. Menghapus nilai-nilai tersebut berisiko menghilangkan informasi yang sangat dibutuhkan untuk membedakan masing-masing tipe anemia. Selain itu, ukuran dataset ini cukup besar, sehingga keberadaan outlier tidak terlalu mendistorsi distribusi data secara keseluruhan. Algoritma yang saya gunakan seperti Random Forest atau XGBoost cukup robust terhadap outlier, sehingga data tetap aman untuk digunakan tanpa perlu melakukan penghapusan.

![image](https://github.com/user-attachments/assets/81800ef8-afa0-4a55-8f10-a0f3d5b0a9c8)

Gambar heatmap di atas menunjukkan korelasi antar fitur numerik dalam dataset darah lengkap (CBC). Beberapa fitur memiliki korelasi yang cukup tinggi, seperti HCT dan MCH (0.61), RBC dan HGB (0.46), serta LYMp dan LYMn yang memiliki korelasi (0,47). Hal ini menunjukkan bahwa fitur-fitur tersebut saling berkaitan dan kemungkinan membawa informasi yang mirip. Sementara itu, sebagian besar fitur lainnya menunjukkan korelasi yang rendah, menandakan bahwa fitur-fitur tersebut relatif independen satu sama lain.

## Data Preparation
Terdapat 3 tahapan dalam persiapan data yaitu
### 1. Encoding fitur kategori dan normalisasi fitur numerik
Dalam dataset ini, fitur "diagnosis" merupakan data kategorikal yang menunjukkan jenis anemia.. Karena sebagian besar algoritma machine learning hanya dapat memproses data numerik, maka fitur ini perlu diubah ke bentuk angka menggunakan teknik Label Encoding. Dengan Label Encoding, masing-masing tipe anemia diberi label numerik secara otomatis. Proses ini membantu model dalam membedakan kelas target tanpa perlu menginterpretasikan nilai-nilai tersebut sebagai urutan.

```python
le = LabelEncoder()
df[categorical_col] = le.fit_transform(df[categorical_col])
```
Hasil label encoding fitur kategorik:

|index|WBC|LYMp|NEUTp|LYMn|NEUTn|RBC|HGB|HCT|MCV|MCH|MCHC|PLT|PDW|PCT|Diagnosis|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|10\.0|43\.2|50\.1|4\.3|5\.0|2\.77|7\.3|24\.2|87\.7|26\.3|30\.1|189\.0|12\.5|0\.17|5|
|1|10\.0|42\.4|52\.3|4\.2|5\.3|2\.84|7\.3|25\.0|88\.2|25\.7|20\.2|180\.0|12\.5|0\.16|5|
|2|7\.2|30\.7|60\.7|2\.2|4\.4|3\.97|9\.0|30\.5|77\.0|22\.6|29\.5|148\.0|14\.3|0\.14|1|
|3|6\.0|30\.2|63\.5|1\.8|3\.8|4\.22|3\.8|32\.8|77\.9|23\.2|29\.8|143\.0|11\.3|0\.12|1|
|4|4\.2|39\.1|53\.7|1\.6|2\.3|3\.93|0\.4|316\.0|80\.6|23\.9|29\.7|236\.0|12\.8|0\.22|5|
 
Algoritma machine learning  sensitif terhadap skala fitur, maka fitur numerik perlu disamakan skala atau distribusinya. Untuk itu, dilakukan standarisasi menggunakan teknik StandardScaler.
StandardScaler mengubah nilai-nilai numerik agar memiliki rata-rata 0 dan deviasi standar 1. Proses ini membantu model dalam memperlakukan setiap fitur secara seimbang tanpa ada fitur yang mendominasi karena skala yang lebih besar.

```python
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```
Hasil standarisasi fitur numerik:

|index|WBC|LYMp|NEUTp|LYMn|NEUTn|RBC|HGB|HCT|MCV|MCH|MCHC|PLT|PDW|PCT|Diagnosis|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|0\.6064337107095714|2\.4500370066013017|-0\.18256124097186915|1\.7803950251995124|-0\.04516356391505246|-0\.6783156977913078|-1\.265679361315967|-0\.20644171267979697|0\.07109262254529823|-0\.052518251791088155|-0\.4890124879935967|-0\.4291365019538645|-0\.6030642953318979|-0\.13117983822675516|5|
|1|0\.6064337107095714|2\.3367558313181225|-0\.16795208011393195|1\.7066927853265677|0\.05848267808554223|-0\.653894441042863|-1\.265679361315967|-0\.1989582629869735|0\.08917405784763383|-0\.05781342980752855|-3\.441460332737944|-0\.5258665173563034|-0\.6030642953318979|-0\.14549723398832556|5|
|2|-0\.18213470758160186|0\.6800186428016393|-0\.11217164774726249|0\.23264798786766633|-0\.2524560479162418|-0\.2596655821036814|-0\.8254511038954594|-0\.14750954634881225|-0\.3158500929246836|-0\.08517184955913717|-0\.6679487210084061|-0\.8697954610094196|-0\.013408460719568938|-0\.1741320255114663|1|
|3|-0\.5200926011349619|0\.6092179082496527|-0\.09357817029170604|-0\.06216097162411402|-0\.45974853191743154|-0\.17244680800209275|-2\.1720316560052475|-0\.12599462848194484|-0\.2833035093804793|-0\.07987667154269681|-0\.5784806045010014|-0\.923534358455219|-0\.9961681850734501|-0\.20276681703460708|1|
|4|-1\.0270294414650019|1\.8694709832750123|-0\.15865534138615367|-0\.2095654513700041|-0\.9779797419204053|-0\.27362058595993566|-3\.0524881708462623|2\.523146562777558|-0\.18566375874786753|-0\.07369896385684971|-0\.6083033100034699|0\.07600913403665001|-0\.5047883228965095|-0\.05959285941890331|5|

### 2. Memisahkan fitur dan target
```python
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
```
Pada tahap ini, data dipisahkan menjadi fitur (X) yang berisi variabel input, seperti hasil pemeriksaan darah, dan target (y) yaitu kolom Diagnosis yang berisi jenis anemia.

### 3 Train test split
```python
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Jumlah data training (X_train): {X_train.shape}')
print(f'Jumlah data testing (X_test): {X_test.shape}')
print(f'Jumlah target training (y_train): {y_train.shape}')
print(f'Jumlah target testing (y_test): {y_test.shape}') 
```
![image](https://github.com/user-attachments/assets/93fa2ed8-bf47-4fdd-a7e9-a6ccead26f72)

Data dibagi 80% untuk pelatihan dan 20% untuk pengujian. Data pelatihan digunakan agar model belajar mengenali pola fitur darah untuk klasifikasi anemia, sedangkan data pengujian untuk mengevaluasi kemampuan model memprediksi anemia pada data baru.

## Modelling
Pada kasus ini, model yang digunakan adalah Random Forest, XGBoost, dan Support Vector Machine (SVM) untuk memprediksi jenis anemia berdasarkan fitur-fitur darah lengkap (CBC). Ketiga algoritma ini dipilih karena memiliki performa tinggi dalam masalah klasifikasi, khususnya pada data medis.

- Random Forest

   Random Forest adalah algoritma ensemble learning berbasis pohon keputusan yang membangun banyak pohon (decision trees) dan menggabungkan hasilnya untuk membuat prediksi yang lebih akurat dan stabil. Random Forest tahan terhadap overfitting dan bekerja baik pada data dengan banyak fitur. Algoritma ini sangat efektif dalam menangani data kategorikal dan numerik tanpa memerlukan banyak praproses (Brownlee, 2020).

- XGBoost

  XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang menggabungkan banyak model pohon secara bertahap. Setiap model baru mempelajari kesalahan dari model sebelumnya untuk meningkatkan akurasi secara bertahap. XGBoost terbukti sangat kuat dalam klasifikasi tabular data, termasuk di bidang kesehatan (Liu, 2018).

- SVM (Support Vector Machine)

  SVM adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane terbaik yang memisahkan kelas-kelas dalam data. SVM efektif pada data berdimensi tinggi dan mampu menangani kasus klasifikasi linier maupun non-linier dengan penggunaan kernel. SVM juga menunjukkan bahwa SVM sering digunakan dalam diagnosis medis karena mampu menghasilkan margin klasifikasi yang maksimum dan akurasi yang tinggi (Liu, 2018). 

Load dan pelatihan model
   - Random Forest
     Menggunakan parameter default umum dengan 100 pohon, tanpa batasan  kedalaman, minimal 2 sampel untuk split, dan random_state=None:
     ```python
     model_rf = RandomForestClassifier(
         n_estimators=100,         
         max_depth=None,           
         min_samples_split=2,      
         random_state=None         
     )
     model_rf.fit(X_train, y_train)
    
     ```
   - XGBoost
     Menggunakan parameter standar dengan penyesuaian agar tidak menggunakan label encoder dan metrik evaluasi log-loss:
     
     ```python
     model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
     model_xgb.fit(X_train, y_train)
     ```
   - SVM (Support Vector Machine)
     Menggunakan kernel RBF, parameter regularisasi default (C=1.0), gamma ‘scale’, dan random_state=None:
     
     ```python
     model_svm = SVC(
         C=1.0,                 
         kernel='rbf',                             
         gamma='scale',                     
         random_state=None        
     )
     model_svm.fit(X_train, y_train)
     ```

## Evaluation
Evaluasi kinerja model dilakukan dengan menggunakan sejumlah metrik utama yaitu: Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix. Metrik ini digunakan karena model bertugas memprediksi jenis anemia, sehingga diperlukan keseimbangan antara kemampuan mendeteksi anemia secara akurat dan menghindari kesalahan klasifikasi.

### Accuracy

  Mengukur seberapa besar persentase prediksi model yang benar dari seluruh prediksi yang dilakukan.
  
  **Rumus** : $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
### Classification Report
  
  1. Precision : Menunjukkan berapa proporsi prediksi positif yang benar-benar sesuai kenyataan.

     $\text{Precision} = \frac{TP}{TP + FP}$
  2. Recall : mengukur seberapa banyak kasus anemia yang berhasil terdeteksi dari semua kasus sebenarnya.

     $\text{Recall} = \frac{TP}{TP + FN}$

     **Keterangan :**
      - **TP**: Prediksi positif yang benar  
      - **TN**: Prediksi negatif yang benar  
      - **FP**: Prediksi positif yang salah  
      - **FN**: Prediksi negatif yang salah

  4. F1-Score : Memberikan gambaran seimbang antara Precision dan Recall. Berguna saat distribusi kelas tidak merata atau saat false positive dan false negative sama-sama berdampak penting.

     $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

Hasil evaluasi tiap model 
- **Random Forest**
  
![image](https://github.com/user-attachments/assets/f135740e-4f94-43b1-ba5f-02a4247b5bdb)

![image](https://github.com/user-attachments/assets/593cfb74-1bf4-4203-aad3-0f21d508758f)

- **XGBoost**

![image](https://github.com/user-attachments/assets/ab0f5a52-19a5-4573-964e-f17ef5744f1d)

![image](https://github.com/user-attachments/assets/df11cd85-808b-48f1-bb8d-4531654b5d03)

- **SVM**
  
![image](https://github.com/user-attachments/assets/2a794674-5fc8-4c00-b208-0835a4700246)

![image](https://github.com/user-attachments/assets/74a5a2cf-43fe-469d-af0d-487614fbfdc9)

Akurasi tiap model
|index|accuracy|
|---|---|
|Random Forest|0\.9919028340080972|
|XGBoost|0\.9919028340080972|
|SVM|0\.7530364372469636|

Sebelum tuning model random forest dan XGBoost memberikan hasil yang sama-sama sangat baik.

### hyperparameter Tuning

```python
# Setup hyperparameter grid dan model
param_grid = {
    'RF': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']},
    'XGB': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
}

estimators = {
    'RF': RandomForestClassifier(),
    'SVM': SVC(),
    'XGB': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Simpan output terbaik
hasil = {}

for nama, clf in estimators.items():
    gs = GridSearchCV(clf, param_grid[nama], cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    acc_cv = cross_val_score(best, X_train, y_train, cv=5).mean()
    acc_test = best.score(X_test, y_test)

    hasil[nama] = {
        'Param Terbaik': gs.best_params_,
        'Akurasi Cross Validation': round(acc_cv, 4),
        'Akurasi Test': round(acc_test, 4),
        'Evaluasi': "Overfit" if acc_cv > acc_test + 0.02 else "Underfit" if acc_test > acc_cv + 0.02 else "Fit"
    }
```

|index|Model|Param Terbaik|Akurasi Cross Validation|Akurasi Test|Evaluasi|
|---|---|---|---|---|---|
|0|RF|\{'max\_depth': 10, 'min\_samples\_split': 2, 'n\_estimators': 200\}|0\.9787|0\.9879|Fit|
|1|SVM|\{'C': 10, 'gamma': 'scale', 'kernel': 'linear'\}|0\.8731|0\.919|Underfit|
|2|XGB|\{'learning\_rate': 0\.1, 'max\_depth': 6, 'n\_estimators': 100\}|0\.9909|0\.9879|Fit|

Akurasi sebelum dan setelah tuning
![image](https://github.com/user-attachments/assets/edf54595-7436-422b-bd2b-b9d9ef3380ca)

## kesimpulan
Dalam proyek klasifikasi jenis anemia ini, tiga algoritma machine learning yang digunakan adalah Random Forest, XGBoost, dan SVM. Sebelum dilakukan tuning hyperparameter, Random Forest dan XGBoost menunjukkan performa yang sangat baik dengan akurasi sebesar 0.9919, sementara SVM tertinggal jauh dengan akurasi 0.7530.

Setelah dilakukan tuning, terjadi beberapa perubahan performa:

- Random Forest menunjukkan sedikit penurunan akurasi test menjadi 0.9879, namun model tetap dalam kategori fit dan memiliki akurasi cross-validation yang baik (0.9787).

- XGBoost juga sedikit menurun akurasi test menjadi 0.9879 dan cross-validation sebesar 0.9909, serta dikategorikan fit.

- SVM mengalami peningkatan signifikan setelah tuning, dengan akurasi test meningkat menjadi 0.919 dari sebelumnya 0.7530. Meskipun begitu, model masih dikategorikan underfit, karena akurasi test lebih tinggi dari cross-validation (0.8731).

Secara keseluruhan, Random Forest dan XGBoost tetap menjadi pilihan terbaik untuk klasifikasi jenis anemia berdasarkan data CBC karena performa tinggi dan konsistensi antara akurasi cross-validation dan test. Di sisi lain, tuning hyperparameter terbukti sangat efektif untuk meningkatkan performa SVM, meskipun hasil akhirnya masih belum menyaingi dua model lainnya.

## Referensi

https://www.who.int/news-room/fact-sheets/detail/anaemia 

Bazgir, O., et al. (2019). Artificial Intelligence in Medicine: Classification of Anemia Types Using Machine Learning Algorithms. *Journal of Medical Systems*, 43(7), 198. 

Khanday, A. M. U. D., et al. (2020). Machine learning techniques for anemia classification. *International Journal of Biomedical Engineering and Technology*, 33(1), 1-20.  

Brownlee, J. (2020). Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python. Machine Learning Mastery. https://books.google.com/books?id=uAPuDwAAQBAJ

Liu, Y. (2018). Python Machine Learning by Example (2nd ed.). Packt Publishing. https://books.google.com/books?id=0nc5DwAAQBAJ


