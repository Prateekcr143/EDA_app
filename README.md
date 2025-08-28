#  Streamlit EDA App

An interactive Exploratory Data Analysis (EDA) web app built with **Streamlit** and **Plotly**.  
Upload your CSV/XLSX dataset and quickly explore schema, summary stats, correlations, missing values, and build custom visualizations.

##  Features
-  Upload CSV/XLSX files
-  Schema & data type overview
-  Summary stats (numeric + categorical)
-  Missing values analysis & handling
-  Data cleaning: drop duplicates, impute missing values, type casting
-  Interactive visual builder (histogram, bar, line, scatter, box, violin, heatmap, pie, sunburst, treemap, area)
-  Download cleaned dataset

##  Run Locally
```bash
# Clone repo
git clone https://github.com/yourusername/streamlit-eda-app.git
cd streamlit-eda-app

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
