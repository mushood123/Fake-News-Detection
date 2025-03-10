
## 📌 Overview

**Fake News Detection** is a **machine learning-based** project designed to classify news articles as **Real or Fake**.  
It leverages **Natural Language Processing (NLP)** techniques to analyze the textual content of news articles and identify misinformation.

🚀 **Key Features**:
- Uses **TF-IDF Vectorization** to convert text into numerical representations.
- Implements a **Random Forest Classifier** for high-accuracy predictions.
- Utilizes **Label Encoding** for categorical feature transformation.
- **Pickle serialization** for model persistence and reusability.
- **Automated preprocessing** (text cleaning, feature engineering).
- **Example prediction demo** with real-world news headlines.

---

## ⚡ Tech Stack

| Technology      | Description |
|---------------|------------|
| **Python**    | Programming language for ML development |
| **Scikit-Learn** | Machine learning library for model training |
| **Pandas**    | Data manipulation and analysis |
| **NumPy**     | Numerical computations |
| **TF-IDF**    | Converts textual data into numerical form |
| **Random Forest** | Classification model for fake news detection |
| **Pickle**    | Saves and loads trained models |
| **Google Colab** | Cloud-based Jupyter Notebook environment |

---

## 📂 Dataset

The model is trained on two datasets:
- 📌 **Fake News Dataset** (`Fake.csv`)
- 📌 **Real News Dataset** (`True.csv`)

**Dataset Structure**:
- `title`: Headline of the news article.
- `text`: Full article text.
- `subject`: Category of news (e.g., politics, world news).
- `date`: Publication date (removed in preprocessing).
- `class`: Label (`Real` or `Fake`).

---

## 🔨 Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
```

### **2️⃣ Install Dependencies**
```sh
pip install pandas numpy scikit-learn
```

### **3️⃣ Load Data (Google Colab)**
```python
from google.colab import drive
drive.mount('/content/drive')

fake_dataset = pd.read_csv("/content/drive/MyDrive/Fake.csv")
real_dataset = pd.read_csv("/content/drive/MyDrive/True.csv")
```

### **4️⃣ Train the Model**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

## 🛠 Model Training Process

✅ **Step 1:** Data Cleaning (Removing punctuation, lowercasing text).  
✅ **Step 2:** Feature Engineering (TF-IDF Vectorization for title & text).  
✅ **Step 3:** Encoding categorical features (`subject` & `class`).  
✅ **Step 4:** Splitting dataset into **80% training & 20% testing**.  
✅ **Step 5:** Training **Random Forest Classifier**.  
✅ **Step 6:** Model Evaluation (Classification Report).  

---

## 📊 Model Evaluation

The trained model is evaluated using the **classification report**, which includes:
- Precision
- Recall
- F1-score
- Accuracy

Example Output:
```sh
              precision    recall  f1-score   support
     Real       0.95       0.93      0.94       1200
     Fake       0.94       0.96      0.95       1300
   Accuracy       0.95       0.95      0.95       2500
```

---

## 🚀 Predicting Fake News

The model allows real-time predictions by transforming user-provided news text.

### **Example Prediction**
```python
example_data = pd.DataFrame({
    'title': ['Grenfell Tower Fire (2017)'],
    'text': ['In June 2017, a fire engulfed the Grenfell Tower in London, killing 72 people. The tragedy exposed serious failures in building safety regulations and sparked widespread outrage and calls for reform'],
    'subject': ['worldnews']
})

predictions = model.predict(example_features)
decoded_predictions = class_encoder.inverse_transform(predictions)
print(decoded_predictions)  # Output: 'Real' or 'Fake'
```

---

## 📦 Model Persistence

To avoid retraining, the trained model and preprocessing objects are **saved using Pickle**:
```python
import pickle

with open('model_and_preprocessing.pkl', 'wb') as f:
    pickle.dump({
        'subject_encoder': subject_encoder,
        'class_encoder': class_encoder,
        'tfidf_vectorizer_title': tfidf_title,
        'tfidf_vectorizer_text': tfidf_text,
        'model': model
    }, f)
```
To **load the model**:
```python
with open('model_and_preprocessing.pkl', 'rb') as f:
    loaded_objects = pickle.load(f)

model = loaded_objects['model']
```

---

## 📬 Contact

👤 **Khawaja Muhammad Mushood**  
📧 **Khawaja.muhammad.mushood@gmail.com**  
🔗 **[GitHub](https://github.com/mushood123/Fake-News-Detection.git)**  

---

⭐ **Star this repo** if you found it useful! 🚀  
