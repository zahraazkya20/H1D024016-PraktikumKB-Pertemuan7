# Praktikum Kecerdasan Buatan - Pertemuan 7
## Jaringan Syaraf Tiruan 2: Klasifikasi Bunga Iris dengan TensorFlow & Keras

**NIM:** H1D024016  
**Nama:** Zahratul Askia  
**Mata Kuliah:** Praktikum Kecerdasan Buatan Shift F  

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan **Jaringan Syaraf Tiruan (JST)** untuk klasifikasi tiga spesies bunga Iris (*Iris Setosa*, *Iris Versicolor*, *Iris Virginica*) menggunakan **TensorFlow** dan **Keras**. Model dilatih dengan 4 fitur morfologi:
- Panjang sepal (sepal length)
- Lebar sepal (sepal width)
- Panjang petal (petal length)
- Lebar petal (petal width)

Output model adalah salah satu dari tiga kelas spesies.

---

## 📁 Struktur Folder
NIM-PraktikumKB-Pertemuan7/  
├── iris.data  
├── main.py  
└── README.md  

---

## Cara Install & Jalankan

### 1. Persyaratan
- Python 3.8–3.11 (TensorFlow belum support Python 3.12+)
- File `iris.data` dalam folder yang sama dengan `main.py`

### 2. Install library
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### 3. Jalankan program
```bash
python main.py
```

