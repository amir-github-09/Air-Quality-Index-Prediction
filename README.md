# 🌫️ Air Quality Prediction Dashboard

This is a complete machine learning project that predicts the **Air Quality Index (AQI)** using a trained **Random Forest Regressor**.  
The app is built using **Streamlit** and includes interactive **EDA dashboards** and a **real-time prediction tool**.

---

## 📌 Project Highlights

✅ Predicts AQI (Air Quality Index) based on air pollutant concentrations  
✅ Built-in Exploratory Data Analysis (EDA) with Seaborn and Matplotlib  
✅ Clean and responsive Streamlit UI  
✅ Final model: **Random Forest Regressor**  
✅ Fully interactive interface with real-time prediction  
✅ Suitable for academic, portfolio, or real-world use

---

## 📂 Dataset

- **Source:** [Air Quality Index](https://www.kaggle.com/datasets/youssefelebiary/air-quality-2024/data)  
- **Target Column:** `AQI`  
- **Features Used:**  
  - CO  
  - CO₂  
  - NO₂  
  - SO₂  
  - O₃  
  - PM2.5  
  - PM10  

---

## 📊 App Features

| Feature             | Description                                                   |
|---------------------|---------------------------------------------------------------|
| 📖 Introduction       | Overview of the project, model, and dataset                  |
| 📈 EDA Dashboard      | Explore statistics, distributions, and correlations          |
| 🔮 Predict AQI        | Enter pollutant values and get live AQI predictions          |
| 🧹 Clean Transitions  | Switching sections clears the previous content automatically |

---

## 🧠 Tools & Models Used

- ✅ Random Forest Regressor  
- ✅ Pandas, Scikit-learn  
- ✅ Seaborn, Matplotlib  
- ✅ Streamlit for building the web UI  
- ✅ Joblib for saving the compressed model and also loading the model  

---

## ▶ How to Run Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the Streamlit app
streamlit run app.py
