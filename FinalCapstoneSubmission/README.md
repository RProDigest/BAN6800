# 🛍️ Takealot Shopper Insights – Behavioral Segmentation & Predictive Modeling

This project analyzes online shopper behavior using a machine learning pipeline incorporating Unsupervised & Supervised Learning:

![Takealot Analytics Framework](image/Takealot_Framework.png)

This six weeks project culminates into a Python Shiny web app that offers a sleek, interactive interface to explore online shopper behavior using machine learning. It leverages **KMeans clustering** to segment customers and **XGBoost** to predict purchasing intent.

🔗 **Live Demo**: [Takealot Shopper Insights App](https://mubanga-nsofu.shinyapps.io/takealot_shopper_insights/)

---

## 📊 Features

- 📁 Upload the provided cleaned `.csv` dataset
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
│   └── xgboost_model.joblib         # Pre-trained XGBoost model
├── requirements.txt                 # Python dependency list
├── image/
│   └── Takealot_Framework.png       # Analytical framework diagram
└── README.md                        # This documentation file
```

---
## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/takealot-shopper-insights.git
cd takealot-shopper-insights

```

---

2. Create a virtual environment (recommended):
``` bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---
3. Install dependencies:

``` bash
pip install -r requirements.txt

```

4. Run the app locally:

```bash

python -m shiny run --reload takealot_app.py

```
Then, open your browser and go to http://127.0.0.1:8000

---
## 🚀  Deployment
This app is deployed via shinyapps.io using rsconnect-python. Deployment steps included:

Model bundling (in /models)

Dependency freezing via requirements.txt

rsconnect CLI used to push to mubanga-nsofu.shinyapps.io

For reproducibility, all models were trained offline and included as .pkl or .joblib files.

---
## 📂  Dataset Usage
The app is designed to work with a cleaned version of the Online Shoppers Intention dataset. This dataset is included in the GitHub repository. After cloning the repository, you can upload the dataset using the app's file upload feature. 
The deployed online model also relies on this same dataset—please ensure you upload the provided CSV file when using the app on shinyapps.io.

---

 
## 🏗️ Architecture Design
🚀 Application Foundation

- ML Model Integration: Pre-loaded KMeans clustering and XGBoost prediction models with SHAP explainer
- Modern UI Framework: Glassmorphic design system with 2025 aesthetic standards
- Modular Architecture: Three distinct functional modules for scalability

📱 Navigation Structure

Tab 1 - **Customer Analysis:** Real-time single customer prediction and segmentation
Tab 2 - **Dataset Analytics:** Batch processing capabilities for large-scale analysis
Tab 3 - **Export Hub:** Comprehensive download center for all analysis outputs

🔬 AI-Powered Analysis Engine

- **12 Behavioral Input Parameters:** Duration metrics, bounce/exit rates, page values, and interaction intensity
- **Dual ML Pipeline:** Simultaneous KMeans clustering and XGBoost probability prediction-
- **Explainable AI:** SHAP analysis for feature importance and model transparency
- **Smart Insights:** Automated recommendation generation based on customer segments

📊 Data Processing Pipeline

**Data Input:** CSV upload functionality with cleaned dataset 
**Interactive Preview:** Real-time data validation and 10-row display preview
**Scalable Batch Analysis:** Configurable sample size control for performance optimization
**Statistical Summaries:** Automated generation of key performance metrics

🎨 Visualization Components

**Dynamic Metric Cards:** Real-time display of segmentation and purchase intent results
**SHAP Feature Plots:** Interactive feature importance visualizations
**Behavior Progress Bars:** Normalized metric displays for quick pattern recognition

💾 Advanced Export System

**Multiple Format Support:** CSV exports for predictions and batch results
**Comprehensive Reports:** Full-text analysis reports with SHAP values and insights
**Timestamped Downloads:** Automated file naming with generation timestamps

🎯 Key Design Principles

**User-Centric Interface:** Intuitive workflow from data input to actionable insights
**Real-Time Responsiveness:** Immediate feedback and dynamic result updates
**Scalable Performance:** Optimized for both single predictions and batch processing
**Professional Output:** Export-ready formats suitable for business reporting

## 📽️ Final Project Requirements (Academic)
This project satisfies Nexford’s final capstone module requirements:

✅ End-to-end machine learning pipeline

✅ Dashboard UI using Shiny for Python

✅ Deployment to cloud (shinyapps.io)

✅ GitHub code versioning

✅ Embedded self-recorded presentation

✅ Interpretability via SHAP

---

## 📢  Author
**Mubanga Nsofu**

*MSc Data Analytics Candidate*
- [GitHub Profile](https://github.com/RProDigest/)
- [Twitter Profile](https://x.com/rprodigest)
- email: mubangansofujr@gmail.com

---

---
## 📝 License
This project is provided for **academic and educational use only**.  
For **extended licensing**, **commercial use**, or integration into proprietary platforms, please contact the author directly.

© 2025 Mubanga Nsofu. All rights reserved.

---
