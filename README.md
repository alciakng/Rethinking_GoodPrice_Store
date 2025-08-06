# Rethinking “Good Price” – A Data-Driven Top-Down Strategy for Local Affordability Policy

This project aims to enhance the **strategic execution of the Good Price Store program** by South Korea’s Ministry of the Interior and Safety.  
By leveraging **regional commercial characteristics** and **unsupervised machine learning clustering**, we analyze affordability trends and propose a **data-driven, top-down policy approach** to support effective designation and activation of Good Price policies.


---

## 🔍 Project Purpose

Current selection methods often rely on static price levels and lack spatial-economic context.  
This project introduces a **systematic clustering approach** that:

- Uses **objective market features** to group regions with similar affordability and commercial patterns  
- Recommends **tailored strategies per cluster type**  
- Promotes **top-down, region-sensitive planning** by local governments

---

## 🔍 Structure of Project 

```
Root
│
├── app.py                     # Main Streamlit app launcher
├── practice.py                # Experimental / development script
├── requirements.txt           # Python dependencies list
├── .gitignore                 # Files and folders to be excluded from Git
│
├── .streamlit/                # Streamlit configuration directory
│
├── controllers/               # Data processing logic modules
│   ├── analysis_cnrtol.py     # Regression, clustering, and strategy analysis
│   └── prepcs_cnrtol.py       # Data preprocessing control functions
│
├── data/                      # Raw and intermediate dataset directory
│
├── exports/                   # Exported results and visualizations
│
├── model/                     # Model outputs and base resources for visualization
│
├── ui/                        # Streamlit UI component modules
│   ├── analysis_board.py      # Analysis results display page
│   ├── chart_board.py         # Visualization and chart board
│   └── main_board.py          # Main dashboard page
│
├── util/                      # Utility functions and configuration files
│   ├── common_util.py         # General-purpose utility functions
│   ├── cluster.json           # Cluster label mapping file
│   └── map.json               # Mapping between Korean administrative names and English labels
│
└── 참고논문/                    # Reference papers and academic documents
```
---

## 🔬 Analytical Pipeline

The project is structured in four stages:

1. **Regression Analysis**  
   Identify and verify the **determinants of Good Price store presence** using panel regression models.

2. **SPC (Supervised Principal Component) Analysis**  
   Reduce feature redundancy and extract **principal components most predictive of policy outcomes**.

3. **Clustering by Principal Components**  
   Segment local markets via unsupervised clustering (KMeans, Hierarchical) based on SPC outputs.

4. **Cluster-Based Strategic Policy Recommendations**  
   Propose distinct **strategic directions for each region type** to support top-down planning.

---

## 📊 Data Overview

The dataset integrates administrative and geospatial information spanning **2020–2024**, including:

- Quarterly store-level panel data  
- **Floating population** (`유동인구수`)  
- **Workplace population** (`직장인구수`)  
- **Resident population** (`상주인구수`)  
- **Number of commercial complexes** (`집객시설수`)  
- **Apartment complex count and average market price** (`아파트 단지수 및 평균 시가`)  
- **Business turnover indicators** (`개업률`, `폐업률`)  
- **Commercial zone change index** (`상권변화지표`)  
- **20s–30s population ratio** (`20~30 인구비`)  

All data is merged based on **official administrative codes** and **shapefile-based boundaries**.

---

## 🗺️ Visualization Highlights

The analysis includes interactive visualization tools:

- **PyDeck maps** for spatial distribution of clusters  
- **Plotly charts** for visualizing Good Price store ratios and trends by cluster  
- **Streamlit dashboard** to explore data dynamically and draw insights

---

## 🛠️ Technologies Used

- **Python:** `pandas`, `geopandas`, `scikit-learn`, `statsmodels`, `prophet`, `pydeck`  
- **Visualization:** `Plotly`, `PyDeck`, `MapboxGL`  
- **Web App Framework:** `Streamlit`  
- **Data Processing:** `GeoJSON`, `Shapefile`, `CSV`

---

## 💡 Contributions

We welcome all feedback and contributions!  
Open an issue or submit a pull request to collaborate or suggest improvements.

---

## 📄 License

This repository is licensed under the **MIT License**.  
Feel free to use, share, and build upon the content with proper attribution.

---

## 🧪 Youtube (Streamlit)

[![Video Label](http://img.youtube.com/vi/HA8PqVclJ5M/0.jpg)](https://youtu.be/HA8PqVclJ5M)


---

## Report

![Image](https://github.com/user-attachments/assets/536979ae-661a-441d-8570-36c298f5d284)
![Image](https://github.com/user-attachments/assets/615b6106-5274-4f96-81a5-19e6f8a94c25)
![Image](https://github.com/user-attachments/assets/ee003945-73df-47a3-8a24-ee302ce80227)
![Image](https://github.com/user-attachments/assets/3f2606dc-7339-46e1-8b99-ae477917a2aa)
![Image](https://github.com/user-attachments/assets/5c1ff902-f838-4039-9db7-a026fe1f58dc)
![Image](https://github.com/user-attachments/assets/1131aab3-0d69-400d-8c41-e156411831db)
![Image](https://github.com/user-attachments/assets/86eacd91-8346-4456-8958-c75644abc0f5)
![Image](https://github.com/user-attachments/assets/59abd845-21ef-4cc2-91e1-0adee4eda102)
![Image](https://github.com/user-attachments/assets/1e9175fa-8afc-491d-af5d-37d202670074)
