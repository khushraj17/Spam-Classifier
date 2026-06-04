# 🛡️ SMS Spam Detector

A Machine Learning-powered web application that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a trained classification model.

## 🚀 Features

- Detects whether an SMS message is Spam or Ham (Not Spam)
- Real-time prediction through an interactive Streamlit interface
- Displays spam probability score
- Text preprocessing using NLP techniques
- Includes sample messages for quick testing
- User-friendly and responsive UI

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Pickle

## 📂 Project Structure

```text
├── app.py
├── model.pkl
├── vectorizer.pkl
├── README.md
└── requirements.txt
````

## 🔍 How It Works

### 1. Text Preprocessing

The input message undergoes several preprocessing steps:

* Convert text to lowercase
* Tokenization
* Remove punctuation
* Remove stopwords
* Apply stemming using Porter Stemmer

### 2. Feature Extraction

The cleaned text is converted into numerical features using a TF-IDF Vectorizer.

### 3. Prediction

The trained machine learning model analyzes the vectorized text and predicts whether the message is:

* Spam
* Not Spam

The application also displays the probability of the message being spam.

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```


## 📈 Machine Learning Pipeline

```text
Input Message
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Trained ML Model
      ↓
Spam / Not Spam Prediction
```

## 🎯 Future Improvements

* Deep Learning-based spam detection
* Email spam classification
* Multi-language support
* Explainable AI predictions
* Model retraining interface
* Deployment on Streamlit Cloud

## 👨‍💻 Author

**Khushraj Rane**

B.Tech (AI & ML) Student

Machine Learning and Artificial Intelligence Enthusiast

## 📄 License

This project is intended for educational and learning purposes.
