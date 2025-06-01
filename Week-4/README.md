
#  Takealot Online Customer Behavioral Segmentation and Intent Prediction

This project analyzes online shopper behavior using a structured machine learning pipeline. It is divided into three main notebooks, each corresponding to a key stage in the data science process:

![Takealot Analytics Framework](Week-4/image/Takealot%20Online%20Customer%20Behavioral%20Segmentation%20and%20Purchasing%20Intent Predictive%20Framework%20Models.png)


This visual framework illustrates the layered approach taken in this project:

### 1. 🧹 Data Cleaning
**Notebook:** `Data_Preparation.ipynb`  
**Key Actions:**
- Handles missing values, duplicates, outliers, and skewness
- Scales and standardizes features
- Encodes categorical variables
- Reduces dimensionality using PCA

### 2. 🧠 Behavioral Segmentation
**Notebook:** `Clustering_Analysis.ipynb`  
**Key Actions:**
- Performs PCA-based dimensionality reduction
- Uses Calinski-Harabasz scores for optimal k
- Applies KMeans clustering
- Profiles clusters based on metrics such as `PageValues`, `BounceRates`, `Interaction_Intensity`
- Evaluates clustering quality using Davies-Bouldin Score

### 3. 🎯 Purchasing Intent Prediction
**Notebook:** `Purchasing_Intent_Classifier.ipynb`  
**Key Actions:**
- Applies XGBoost and Logistic Regression with hyperparameter tuning (Optuna)
- Handles class imbalance using SMOTE 
- Visualizes model performance with ROC and Precision-Recall Curves
- Applies SHAP for interpretability
- Assesses overfitting using train/test metrics and cross-validation
- Predicts whether a user will generate revenue (i.e., make a purchase).
**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall Curves
- SHAP-based interpretability of predictions



### 4. 📁 How to Run This Project

4.1. Clone the repository:
   ```bash
   git clone https://github.com/rprodigest/Week-4.git
   cd Week-4
   ```
4.2. 📎 Requirements
- Python 3.8+, pandas, numpy, matplotlib
- seaborn, SweetViz, scikit-learn, optuna 
- shap, joblib, xgboost, imbalance

4.3 If any requirements are missing simply install using pip command:
```python
e.g. pip install pandas

```


4.3. Place the dataset online_shoppers_cleaned.csv in the project root directory.

4.4 Run the notebooks in the following order:

 - Data_Preparation.ipynb

- Clustering_Analysis.ipynb

- Purchasing_Intent_Classifier.ipynb



### 5 📦 Artifacts
📊 xgboost_model.joblib – Saved model for deployment

📈 SHAP summary plots for model explainability

📉 ROC and PRC visualizations

🧾 Classification reports for both XGBoost and Logistic Regression


### 6 📌 Project Highlights
✅ Integrated unsupervised and supervised learning

✅ Model interpretability using SHAP

✅ Hyperparameter tuning via Optuna

✅ Overfitting mitigation using stratified CV and early stopping
## Author
- Mubanga Nsofu (Nexford University)

## Contacts
- [@rprodigest](https://x.com/rprodigest)
- mubangansofujr@gmail.com

