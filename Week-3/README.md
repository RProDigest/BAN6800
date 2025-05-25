
#  Week 3 – Dataset Preparation and Exploratory Data Analysis
Online Shoppers Purchasing Intention dataset preparation and exloratory data analysis (EDA)

# Abstract

The dataset is from UCI Machine Learning Repository, consisting of 12,330 samples with 10 numerical attributes and 8 categorical attributes. The project focuses on:

> **"Behavioral Segmentation and Predictive Modeling of Purchasing Intent Among Takealot Online Shoppers."**
## Author
- Mubanga Nsofu (Nexford University)

## Contacts
- [@rprodigest](https://x.com/rprodigest)
- mubangansofujr@gmail.com


## 📎 Requirements
- Python 3.8+
- pandas, matplotlib, seaborn, SweetViz


## ▶️ How to Run the Notebook

To run this notebook locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RProDigest/BAN6800.git
   cd BAN6800/week-3

2. **Dataset Location**: Place the dataset file (e.g., online_shoppers_intention.csv) into the week-3/ folder — this is the same folder where the notebook is located.

3. **Install dependencies (if not already installed)**:

```python
pip install pandas matplotlib seaborn jupyter SweetViz
```
4. **Launch Jupyter Notebook**:

```python
jupyter notebook
```

5. **In your browser, open the file:**

- MilestoneOne Assignment_BAN6800_Mubanga_Nsofu.ipynb
- Then click Run All or execute cells step-by-step.


## 📊 Key Steps Performed

### ✅ 1. Data Cleaning
- Removed missing and duplicate entries
- Encoded categorical variables, including `VisitorType` using one-hot encoding
- Converted Boolean values (e.g., `Weekend`, `Revenue`) to integers

### ✅ 2. Feature Engineering
- Created `Total_Duration` (sum of all duration columns)
- Created `Interaction_Intensity` to capture time-per-product engagement

### ✅ 3. Exploratory Data Analysis (EDA)
- Visualized distributions for features like `PageValues`, `BounceRates`, and `Interaction_Intensity`
- Used boxplots to highlight relationships between user behavior and purchase outcomes
- Identified class imbalance in the `Revenue` target variable

## 📂 Folder Contents
- `original dataset`: dataset before preprocessing called `online_shoppers_intention.csv`
- `cleaned_data.csv`: Final dataset after preprocessing called `online_shoppers_intention_cleaned_file.csv`
- `MilestoneOne Assignment_BAN6800_Mubanga_Nsofu.ipynb`: Main notebook
- visuals: Sample output plots## 📌 Notes
- No machine learning models were applied yet — this notebook is focused on preparing the dataset for segmentation and predictive modeling in future milestones.
- The final dataset is ready for unsupervised learning (e.g., clustering & PCA) and classification tasks.
 ## 🧠 Next Steps
- Apply PCA and clustering to segment user behavior
- Build supervised models to predict `Revenue` (purchase intent)
- Deploy solution in the cloud



