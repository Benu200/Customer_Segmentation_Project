# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:49:47 2025

@author: ADMIN
"""

# -------------------------------------------------------------
# 🧩 CUSTOMER SEGMENTATION DASHBOARD USING K-MEANS CLUSTERING
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Streamlit Page Setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧩 Customer Segmentation Dashboard")

# -------------------------------------------------------------
# 1️⃣ Load the Dataset
# -------------------------------------------------------------
file_path = r"E:\jd\segmented_customers.csv"   # ✅ Your exact file path

try:
    df = pd.read_csv(file_path)
    st.success("✅ Data loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ File not found at: {file_path}")
    st.stop()

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------------------
# 2️⃣ Data Cleaning
# -------------------------------------------------------------
df.columns = df.columns.str.strip()  # remove extra spaces

# Fill missing numeric values with median
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

st.success("✅ Data cleaning completed!")

# -------------------------------------------------------------
# 3️⃣ Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
st.header("🔍 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribution of Annual Income")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df['Annual Income'], kde=True, color='skyblue', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📦 Spending Score Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(y=df['Spending Score'], color='lightgreen', ax=ax)
    st.pyplot(fig)

st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# -------------------------------------------------------------
# 4️⃣ Feature Scaling
# -------------------------------------------------------------
st.header("⚙️ Feature Scaling & Model Preparation")
features = ['Age', 'Annual Income', 'Spending Score']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

st.success("✅ Features scaled successfully!")

# -------------------------------------------------------------
# 5️⃣ Determine Optimal K using Elbow & Silhouette Methods
# -------------------------------------------------------------
st.header("📊 Determine Optimal Number of Clusters")

inertia = []
sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_scaled, labels))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Elbow Method")
    fig, ax = plt.subplots()
    plt.plot(range(2, 11), inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    st.pyplot(fig)

with col2:
    st.subheader("Silhouette Score")
    fig, ax = plt.subplots()
    plt.plot(range(2, 11), sil_scores, marker='o', color='red')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    st.pyplot(fig)

optimal_k = st.slider("Select Optimal Number of Clusters (K)", 2, 10, 3)

# -------------------------------------------------------------
# 6️⃣ Apply K-Means Clustering
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

st.success(f"✅ K-Means applied with K = {optimal_k}")

st.subheader("📍 Cluster Counts")
st.bar_chart(df['Cluster'].value_counts())

# -------------------------------------------------------------
# 7️⃣ PCA Visualization (2D)
# -------------------------------------------------------------
st.header("🎨 PCA Visualization of Clusters")

pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=components[:,0], y=components[:,1],
                hue=df['Cluster'], palette='viridis', s=70)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Customer Segments (PCA Visualization)")
st.pyplot(fig)

# -------------------------------------------------------------
# 8️⃣ Cluster Profiling
# -------------------------------------------------------------
st.header("📋 Cluster Profiling")

cluster_profile = df.groupby('Cluster')[features].mean().round(2)
st.dataframe(cluster_profile)

st.markdown("""
- 🟢 **Cluster 0**: High income, high spending customers  
- 🟣 **Cluster 1**: Moderate income and spending  
- 🟡 **Cluster 2**: Low income, low spending customers
""")

# -------------------------------------------------------------
# ✅ Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown("Developed with ❤️ using **Streamlit**, **Scikit-learn**, and **Pandas**.")
