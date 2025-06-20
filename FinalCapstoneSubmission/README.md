# 🛍️ Takealot Shopper Insights – Behavioral Segmentation & Predictive Modeling

This Python Shiny web app offers a sleek, interactive interface to explore online shopper behavior using machine learning. It leverages **KMeans clustering** to segment customers and **XGBoost** to predict purchasing intent.

🔗 **Live Demo**: [Takealot Shopper Insights App](https://mubanga-nsofu.shinyapps.io/takealot_shopper_insights/)

---

## 📊 Features

- 📁 Upload your cleaned `.csv` dataset
- 📌 Segment customers using **KMeans Clustering**
- 🔮 Predict purchase intent using **XGBoost Classifier**
- 🧠 SHAP analysis for feature importance
- 📉 Clean visualizations and user-friendly layout
- 🖼️ Modern UI with two-tab navigation and Takealot branding

---

## 🤖 Models Used

| Task            | Algorithm | Purpose                        |
|-----------------|-----------|--------------------------------|
| Clustering      | KMeans    | Behavioral segmentation        |
| Classification  | XGBoost   | Purchase intent prediction     |

---

## 📁 Folder Structure

```plaintext
.
├── takealot_app.py                  # Main Shiny app script
├── models/
│   ├── kmeans_model.pkl             # Pre-trained KMeans model
│   └── xgboost_model.pkl            # Pre-trained XGBoost model
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation file
