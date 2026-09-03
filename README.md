# 🏦 Credit Risk Prediction System

An end-to-end **Machine Learning–based Credit Risk Prediction application** that evaluates the likelihood of loan default using applicant financial and demographic data. This project demonstrates a complete ML workflow — from data preprocessing and model training to deployment via an interactive **Streamlit web application**.

---

## 🚀 Key Features
- End-to-end Machine Learning pipeline
- Handles class imbalance using **SMOTE**
- Hyperparameter tuning with **GridSearchCV**
- Logistic Regression model with strong interpretability
- Model evaluation using **ROC-AUC**, confusion matrix, and classification report
- Interactive **Streamlit UI** for real-time predictions
- Gauge chart visualization for credit risk level
- Cloud deployment ready (Streamlit Cloud / Render)

---

## 🧠 Machine Learning Workflow
1. Data loading and exploration  
2. Missing value handling  
3. Feature scaling & one-hot encoding  
4. Class imbalance handling (SMOTE)  
5. Model training (Logistic Regression)  
6. Hyperparameter tuning  
7. Model evaluation  
8. Model persistence using Joblib  
9. Frontend integration using Streamlit  

---

## 📂 Project Structure

---

## ⚙️ Tech Stack
- **Programming Language:** Python 3.9  
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn  
- **Visualization:** Plotly, Seaborn, Matplotlib  
- **Frontend:** Streamlit  
- **Model Storage:** Joblib  

---

## 📊 Model Details
- **Algorithm:** Logistic Regression  
- **Imbalance Handling:** SMOTE  
- **Hyperparameter Tuning:** GridSearchCV (5-Fold CV)  
- **Evaluation Metric:** ROC-AUC  

---📈 Application Highlights

User-friendly sliders and dropdowns for applicant input

Real-time credit risk probability score

Risk level visualization using a gauge chart

Clear classification: High Risk / Low Risk

Suitable for fintech and banking use cases

📌 Use Cases

Credit scoring systems

Loan approval decision support

Financial risk assessment

Fintech portfolio projects

🛠️ Future Improvements

Support for additional ML models (Random Forest, XGBoost)

Model explainability using SHAP or LIME

Database integration for storing predictions

Authentication and role-based access

API deployment using FastAPI

👨‍💻 Author

Gursehaj Singh
Computer Science Undergraduate (CSE ’26)
Interests: Software Development, Artificial Intelligence, Data Analytics

## 🏗️ Environment Setup

### 1️⃣ Create Project Environment
```bash
conda create -n credit_risk python=3.9 -y
conda activate credit_risk
python train_model.py
streamlit run app.py
LIVE DEMO : https://drive.google.com/file/d/1bzLH5QmHyKYWf-k66gr1KwfZRn08wGQf/view?usp=sharing
