# 🍊🍇 UTS - Machine Learning: Fruit Classifier (Naive Bayes)

Proyek ini bertujuan untuk membangun model klasifikasi buah, khususnya untuk membedakan antara **jeruk (orange)** dan **anggur (grapefruit)** menggunakan algoritma **Naive Bayes**. Dataset yang digunakan berisi fitur-fitur karakteristik dari buah-buahan tersebut.

---

## 📁 Dataset
Dataset:*https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit*
Jumlah sampel: 10.000 baris  
Fitur: Karakteristik buah (seperti diameter, weight, min/max pixel, dsb)  
Label: `name` (orange atau grapefruit)

---

## 🔍 Tahapan dan Langkah-Langkah

### 1. Import Library
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### 2. Load Dataset
```python
df = pd.read_csv('citrus.csv')
```

---

### 3. Eksplorasi Data
- Melihat struktur data dengan `df.info()` dan `df.describe()`
- Memeriksa distribusi kelas:
```python
sns.countplot(x='name', data=df)
```

---

### 4. Preprocessing
- Pisahkan fitur dan label
```python
X = df.drop('name', axis=1)
y = df['name']
```

- Split data (80% train, 20% test)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

### 5. Pelatihan Model (Naive Bayes)
```python
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
```

---

### 6. Evaluasi Model
```python
y_pred = nb_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))
```

📊 **Hasil Evaluasi**:
- **Akurasi**: `92.6%`
- **Precision, Recall, F1-Score** untuk kedua kelas ≈ 0.93
- Model cukup seimbang dan efisien untuk klasifikasi 2 kelas

---

## 📈 Confusion Matrix
```
           Predicted
           G   O
Actual G  933  67
       O   81 919
```

---

## 📝 Catatan Tambahan
- Model ini hanya menggunakan algoritma **Naive Bayes (GaussianNB)**.
- Untuk peningkatan performa bisa dibandingkan dengan algoritma lain seperti SVM atau Random Forest.
- Dataset dalam kondisi bersih, tidak ada null value.

---
