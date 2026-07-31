# 🤖 AI Sentiment Chatbot using Sentiment140

## 📌 Project Overview

The AI Sentiment Chatbot is a Natural Language Processing (NLP) application that analyzes the sentiment of user input using a deep learning model trained on the Sentiment140 Twitter dataset. Based on the detected sentiment, the chatbot provides an appropriate response through an interactive Streamlit web interface.

---

## 🚀 Features

- Sentiment Analysis using LSTM
- Interactive Chatbot Interface
- Built with Streamlit
- Trained on the Sentiment140 Twitter Dataset
- Predicts Positive or Negative Sentiment
- Displays Confidence Score
- Responsive and User-Friendly UI

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- LSTM (Long Short-Term Memory)
- Natural Language Processing (NLP)
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## 📂 Dataset

**Dataset:** Sentiment140 Twitter Dataset

- 1.6 Million Tweets
- Binary Sentiment Classification
- Labels:
  - 0 → Negative
  - 4 → Positive

---

## 📁 Project Structure

```
SentimentChatbot/
│── app.py
│── style.css
│── requirements.txt
│── sentiment_model.keras
│── tokenizer.pkl
│── README.md
```

---

## ⚙️ Installation

Clone or download the project.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 🧠 Working

1. User enters a message.
2. The text is preprocessed using NLP techniques.
3. The tokenizer converts text into numerical sequences.
4. The LSTM model predicts the sentiment.
5. The chatbot displays:
   - Predicted Sentiment
   - Confidence Score
   - Appropriate Response

---

## 📸 Sample Inputs

### Positive

```
I love this movie.
```

Output:

```
Positive 😊
```

---

### Negative

```
I hate this movie.
```

Output:

```
Negative 😔
```

---

## 📚 NLP Techniques Used

- Text Cleaning
- Lowercasing
- URL Removal
- Mention Removal
- Tokenization
- Sequence Padding
- Sentiment Classification using LSTM

---

## 👨‍💻 Developed By

**Palak Narang**

B.Tech Computer Science and Engineering

VIT Bhopal University

---

## 📄 License

This project is developed for educational and academic purposes.
