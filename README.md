# 💳 Credit Card Fraud Detection (Machine Learning + Streamlit)

## 📌 Project Overview
This project detects fraudulent credit card transactions using **anomaly detection** and a **supervised classifier**.  
It is built using **Python, Scikit-Learn, XGBoost, Pandas, and Streamlit**.  

The workflow includes:
- Data preprocessing & balancing  
- Anomaly detection with **Isolation Forest** & **Local Outlier Factor (LOF)**  
- Fraud classification with **XGBoost**  
- Model evaluation (Confusion Matrix, ROC Curve)  
- Interactive **Streamlit dashboard** for predictions  

Dataset used: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

---

## ⚙️ Tech Stack
- Python 3.8+  
- Pandas, NumPy  
- Scikit-Learn  
- XGBoost  
- Matplotlib, Seaborn  
- Streamlit  

---

## 🚀 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/creditcard_fraud_project.git
   cd creditcard_fraud_project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
   - Place it inside the project folder (same level as the notebook).  

---

## 🧪 Model Training
Run the Jupyter notebook:
```bash
jupyter notebook fraud_detection_creditcard.ipynb
```
This will:
- Train Isolation Forest, LOF, and XGBoost  
- Save trained model and artifacts inside `artifacts/`  
- Generate confusion matrix and ROC curve  

---

## 📊 Streamlit Dashboard
Run the app:
```bash
streamlit run app_streamlit.py
```

### Features
- **Predict Tab** → Upload a CSV (`Time, V1..V28, Amount`) → get fraud predictions  
- **Metrics Tab** → View confusion matrix & ROC curve

- 
  ## 📷 Screenshots

<img width="1844" height="1140" alt="image" src="https://github.com/user-attachments/assets/b5444e92-9ce7-4ec5-a966-8e831db885d3" />
<img width="1675" height="968" alt="image" src="https://github.com/user-attachments/assets/b11fd8f6-2240-4a45-a459-0c0df3828c0c" />

## 🙌 Acknowledgements
- Dataset by **Machine Learning Group (ULB)** – [Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  


