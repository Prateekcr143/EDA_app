"""
 Streamlit EDA App
===========================

Features
--------
- Upload CSV/XLSX files
- Quick EDA: schema, summary stats, missing values, correlations
- Visual Builder: pick chart types & columns (Plotly, interactive)
- Data Cleaning: drop duplicates, impute missing values, type casting

Run Locally
-----------
1) Create a virtual environment (optional but recommended)
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

2) Install dependencies
   #  minimal:
   pip install streamlit pandas numpy plotly openpyxl scikit-learn

3) Run the app
   streamlit run streamlit_ai_eda_app.py
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="EDA App", layout="wide")
st.title(" EDA App (Upload data + Explore)")

# -----------------------------
# Utilities
# -----------------------------

def _is_numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col)

@st.cache_data(show_spinner=False)
def load_file(file: bytes, filename: str) -> pd.DataFrame:
    suffix = filename.lower().split(".")[-1]
    if suffix in ("csv"):
        return pd.read_csv(io.BytesIO(file))
    if suffix in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file))
    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")

# -----------------------------
# Sidebar - Data Ingestion
# -----------------------------
with st.sidebar:
    st.header(" Data")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv" , "xlsx", "xls"], accept_multiple_files=False)
    # st.caption("Upload your dataset (no sample datasets included)")
    load_btn = st.button("Load Data")

if load_btn:
    st.session_state.pop("df", None)

df = None
if uploaded is not None and ("df" not in st.session_state or load_btn):
    try:
        df = load_file(uploaded.getvalue(), uploaded.name)
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    df = st.session_state.get("df")

if df is None:
    st.info(" Upload a dataset to begin.")
    st.stop()

st.success(f"Loaded dataset with shape: {df.shape}")

# -----------------------------
# Tabs
# -----------------------------
Overview, Clean, Visualize = st.tabs(["Overview", "Clean", "Visualize"]) 

# -----------------------------
# Overview Tab
# -----------------------------
with Overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Schema & Types")
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [df[c].notna().sum() for c in df.columns],
        "nulls": [df[c].isna().sum() for c in df.columns],
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    st.dataframe(schema, use_container_width=True)

    st.subheader("Summary Statistics (numeric)")
    if df.select_dtypes(include=np.number).shape[1] > 0:
        st.dataframe(df.describe().T, use_container_width=True)
    else:
        st.caption("No numeric columns detected.")

    st.subheader("Missingness")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_df = miss[miss > 0].rename_axis("column").reset_index(name="missing")
    if not miss_df.empty:
        fig = px.bar(miss_df, x="column", y="missing", title="Missing values by column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No missing values detected.")

    st.subheader("Correlation (numeric)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap (numeric)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Need at least two numeric columns for correlation.")

# -----------------------------
# Clean Tab
# -----------------------------
with Clean:
    st.subheader("Data Cleaning")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Duplicates**")
        if st.button("Drop duplicate rows"):
            before = len(df)
            df.drop_duplicates(inplace=True)
            st.session_state["df"] = df
            st.success(f"Removed {before - len(df)} duplicate rows.")

        st.markdown("**Missing Values**")
        target_cols = st.multiselect("Select columns to impute/drop", df.columns.tolist())
        strategy = st.selectbox("Strategy", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with constant"])
        const_val = None
        if strategy == "Fill with constant":
            const_val = st.text_input("Constant value (string will be cast if possible)")
        if st.button("Apply missing-value strategy") and target_cols:
            if strategy == "Drop rows":
                df.dropna(subset=target_cols, inplace=True)
            elif strategy == "Fill with mean":
                for col in target_cols:
                    if _is_numeric(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "Fill with median":
                for col in target_cols:
                    if _is_numeric(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "Fill with mode":
                for col in target_cols:
                    if df[col].mode(dropna=True).size:
                        df[col].fillna(df[col].mode(dropna=True)[0], inplace=True)
            elif strategy == "Fill with constant":
                for col in target_cols:
                    df[col].fillna(const_val, inplace=True)
            st.session_state["df"] = df
            st.success("Missing-value operation applied.")

    with c2:
        st.markdown("**Type Casting**")
        cast_col = st.selectbox("Column", ["(choose)"] + df.columns.tolist())
        cast_to = st.selectbox("To type", ["int", "float", "str", "datetime"], index=2)
        if st.button("Cast type") and cast_col != "(choose)":
            try:
                if cast_to == "int":
                    df[cast_col] = pd.to_numeric(df[cast_col], errors="coerce").astype("Int64")
                elif cast_to == "float":
                    df[cast_col] = pd.to_numeric(df[cast_col], errors="coerce")
                elif cast_to == "str":
                    df[cast_col] = df[cast_col].astype(str)
                elif cast_to == "datetime":
                    df[cast_col] = pd.to_datetime(df[cast_col], errors="coerce")
                st.session_state["df"] = df
                st.success(f"Casted '{cast_col}' to {cast_to}.")
            except Exception as e:
                st.error(f"Failed to cast: {e}")

    st.download_button(
        " Download Cleaned Data ",
        df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )





# -----------------------------
# Visualize Tab (Extended)
# -----------------------------
with Visualize:
    st.subheader("Visual Builder")
    chart_type = st.selectbox(
        "Chart Type", 
        [
            "Histogram", "Bar", "Line", "Scatter", "Box", "Heatmap (correlation)",
            "Pie", "Donut", "Area", "Violin", "Sunburst", "Treemap"
    ]
        
    
)

    

    cols = df.columns.tolist()
    x = st.selectbox("X", [None] + cols)
    y = st.selectbox("Y", [None] + cols)
    color = st.selectbox("Color / Group (optional)", [None] + cols)
    agg = st.selectbox("Aggregation (for Bar/Line/Area)", ["count", "sum", "mean", "median", "min", "max"]) if chart_type in ("Bar", "Line", "Area") else None

    if st.button("Create Chart"):
        try:
            if chart_type == "Histogram":
                fig = px.histogram(df, x=x, color=color, title=f"Histogram of {x}")
            elif chart_type == "Bar":
                d = df.groupby(x)[y].agg(agg).reset_index()
                fig = px.bar(d, x=x, y=y, color=color if color in d.columns else None, title=f"Bar: {agg}({y}) by {x}")
            elif chart_type == "Line":
                d = df.groupby(x)[y].agg(agg).reset_index()
                fig = px.line(d, x=x, y=y, color=color if color in d.columns else None, title=f"Line: {agg}({y}) by {x}")
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {y} vs {x}")
            elif chart_type == "Box":
                fig = px.box(df, x=x, y=y, color=color, title=f"Box plot: {y} by {x}")
            elif chart_type == "Heatmap (correlation)":
                num_cols = df.select_dtypes(include=np.number).columns
                corr = df[num_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            elif chart_type == "Pie":
                fig = px.pie(df, names=x, values=y, color=color, title=f"Pie chart of {y} by {x}")
            elif chart_type == "Donut":
                fig = px.pie(df, names=x, values=y, hole=0.4, color=color, title=f"Donut chart of {y} by {x}")
            elif chart_type == "Area":
                d = df.groupby(x)[y].agg(agg).reset_index()
                fig = px.area(d, x=x, y=y, color=color if color in d.columns else None, title=f"Area chart: {agg}({y}) by {x}")
            elif chart_type == "Violin":
                fig = px.violin(df, x=x, y=y, color=color, box=True, points="all", title=f"Violin plot: {y} by {x}")
            elif chart_type == "Sunburst":
                fig = px.sunburst(df, path=[x, color] if color else [x], values=y, title="Sunburst Chart")
            elif chart_type == "Treemap":
                fig = px.treemap(df, path=[x, color] if color else [x], values=y, title="Treemap Chart")
            st.plotly_chart(fig, use_container_width=True)
            
           
        except Exception as e:
            st.error(f"Could not create chart: {e}")
