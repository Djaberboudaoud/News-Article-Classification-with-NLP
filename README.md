![BBC News Classification Banner](https://raw.githubusercontent.com/your-username/your-repo/main/images/banner.png)

# 📰 BBC News Classification

A **machine learning project** that automatically classifies BBC news articles into categories using **LSTM Neural Networks** and **Naive Bayes** models.

---

## 📌 Overview

This project focuses on **text classification** of BBC news articles into 5 categories:

- 💼 **Business**  
- 🎭 **Entertainment**  
- 🏛️ **Politics**  
- 🏅 **Sport**  
- 💻 **Tech**

The system combines **deep learning (LSTM)** for advanced sequence modeling with **Multinomial Naive Bayes** for a fast and efficient traditional ML baseline.

---

## 🧠 Models Used

| Model                       | Description                                   | Key Feature                        |
|-----------------------------|-----------------------------------------------|-------------------------------------|
| **LSTM Neural Network**     | Deep learning approach using word embeddings | Captures sequential dependencies   |
| **Multinomial Naive Bayes** | Traditional ML with TF-IDF                   | Fast and performs well on text data |

---

## 🛠️ Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn
pip install nltk tensorflow scikit-learn joblib
