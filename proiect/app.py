import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm

# Configurare pagină
st.set_page_config(page_title="Analiză McDonald's + Prezentare Cod", layout="wide")
st.title("📊 Analiză McDonald's și Prezentare Cod")

st.markdown("""
Această aplicație include:
- Analize financiare pentru McDonald's bazate pe fișiere CSV.
- O prezentare a codului organizată în secțiuni pentru a explica logica din spatele fiecărui pas.
""")

# Meniu lateral pentru selectarea secțiunii de prezentare
section = st.sidebar.radio("Selectează secțiunea de prezentare", [
    "1. Încărcare și preprocesare date",
    "2. Tratare valori lipsă",
    "3. Analize financiare",
    "4. Regresie multiplă",
    "5. Clusterizare cu KMeans",
    "6. Alte analize"
])

# Afișăm codul funcțional al proiectului (exemplificat pe secțiuni)
if section == "1. Încărcare și preprocesare date":
    with st.expander("Cod: Încarcă datele"):
        st.code(
            """
@st.cache_data
def load_data():
    # Citește fișierul CSV din folderul 'proiect'
    mcd = pd.read_csv("proiect/McDonalds_financial_statements.csv", index_col=0)
    
    # Înlocuirea valorilor lipsă (numeric cu medie, non-numeric cu mod)
    def nan_replace(tabel):
        for var in tabel.columns:
            if tabel[var].isna().any():
                if is_numeric_dtype(tabel[var]):
                    tabel[var].fillna(tabel[var].mean(), inplace=True)
                else:
                    tabel[var].fillna(tabel[var].mode()[0], inplace=True)
    nan_replace(mcd)
    
    # Convertim indexul la int
    mcd.index = mcd.index.astype(int)
    return mcd

@st.cache_data
def load_gdp():
    return pd.read_csv("proiect/gdp.csv")

mcd = load_data()
gdp = load_gdp()
            """,
            language="python"
        )
    st.write("Datele au fost încărcate și preprocesate cu succes.")

elif section == "2. Tratare valori lipsă":
    with st.expander("Cod: Tratare valori lipsă"):
        st.code(
            """
# Verificăm valorile lipsă după imputare și le afișăm
st.write("Număr de valori lipsă pe coloană:")
st.dataframe(mcd.isna().sum().to_frame("missing"))
            """,
            language="python"
        )
    st.dataframe(mcd.isna().sum().to_frame("missing"))

elif section == "3. Analize financiare":
    with st.expander("Cod: Analiza Raport Datorii/Active și Creștere Venituri"):
        st.code(
            """
# Problema 1: Calculul raportului datoriilor la active
ratios = []
for year, row in mcd.iterrows():
    td = row['Total debt ($B)']
    ta = row['Total assets ($B)']
    if ta != 0:
        ratios.append((year, td / ta))
df_ratio = pd.Series({y: r for y, r in ratios})
min_year = df_ratio.idxmin()
st.line_chart(df_ratio, height=300)
st.write(f"Anul cu cel mai mic raport Debt/Assets: {min_year}")

# Problema 2: Creștere procentuală anuală a veniturilor
rev = mcd['Revenue ($B)']
growth = rev.pct_change() * 100
max_year = growth.idxmax()
st.bar_chart(growth.dropna(), height=300)
st.write(f"Anul cu cea mai mare creștere procentuală: {max_year} ({growth.max():.2f}%)")
            """,
            language="python"
        )
    st.markdown("### Raport Datorii/Active")
    ratios = []
    for year, row in mcd.iterrows():
        td = row['Total debt ($B)']
        ta = row['Total assets ($B)']
        if ta != 0:
            ratios.append((year, td / ta))
    df_ratio = pd.Series({y: r for y, r in ratios})
    st.line_chart(df_ratio, height=300)
    st.write(f"Anul cu cel mai mic raport Debt/Assets: **{df_ratio.idxmin()}**")
    
    st.markdown("### Creștere Venituri")
    rev = mcd['Revenue ($B)']
    growth = rev.pct_change() * 100
    st.bar_chart(growth.dropna(), height=300)
    st.write(f"Anul cu cea mai mare creștere procentuală: **{growth.idxmax()}** ({growth.max():.2f}%)")

elif section == "4. Regresie multiplă":
    with st.expander("Cod: Regresie multiplă Revenue ~ MarketCap + Earnings + Debt"):
        st.code(
            """
# Selectăm variabilele pentru regresie și adăugăm constantă
X = mcd[['Market cap ($B)', 'Earnings ($B)', 'Total debt ($B)']]
X = sm.add_constant(X)
y = mcd['Revenue ($B)']

# Construim și potrivim modelul OLS
model = sm.OLS(y, X, missing='drop').fit()
st.text(model.summary().as_text())
            """,
            language="python"
        )
    X = mcd[['Market cap ($B)', 'Earnings ($B)', 'Total debt ($B)']]
    X = sm.add_constant(X)
    y = mcd['Revenue ($B)']
    model = sm.OLS(y, X, missing='drop').fit()
    st.text(model.summary().as_text())

elif section == "5. Clusterizare cu KMeans":
    with st.expander("Cod: Clusterizare KMeans"):
        st.code(
            """
# Alegem variabilele de interes, standardizăm datele
features = ['Market cap ($B)', 'Revenue ($B)', 'Earnings ($B)',
            'P/E ratio', 'Operating Margin (%)', 'Net assets ($B)',
            'Total assets ($B)', 'Total debt ($B)']
data = (mcd[features] - mcd[features].mean()) / mcd[features].std()

# Selectăm numărul de clustere
k = st.slider("Număr de clustere", 2, 6, 3)
km = KMeans(n_clusters=k, random_state=0).fit(data)
mcd['Cluster'] = km.labels_

# Afișăm rezultatele clusterizării
st.dataframe(mcd[['Cluster']])
fig, ax = plt.subplots(figsize=(10, 4))
for c in range(k):
    sel = mcd[mcd['Cluster'] == c]
    ax.scatter(sel.index, sel['Market cap ($B)'], label=f'Cluster {c}')
ax.set_xlabel("An")
ax.set_ylabel("Market cap ($B)")
ax.legend()
st.pyplot(fig)
            """,
            language="python"
        )
    # Codul efectiv pentru clusterizare:
    features = ['Market cap ($B)', 'Revenue ($B)', 'Earnings ($B)',
                'P/E ratio', 'Operating Margin (%)', 'Net assets ($B)',
                'Total assets ($B)', 'Total debt ($B)']
    data = (mcd[features] - mcd[features].mean()) / mcd[features].std()
    k = st.slider("Număr de clustere", 2, 6, 3, key="clusters")
    km = KMeans(n_clusters=k, random_state=0).fit(data)
    mcd['Cluster'] = km.labels_
    st.dataframe(mcd[['Cluster']])
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in range(k):
        sel = mcd[mcd['Cluster'] == c]
        ax.scatter(sel.index, sel['Market cap ($B)'], label=f'Cluster {c}')
    ax.set_xlabel("An")
    ax.set_ylabel("Market cap ($B)")
    ax.legend()
    st.pyplot(fig)

elif section == "6. Alte analize":
    with st.expander("Cod: Top 3 ani profit, Categorizare ani, Calcul ROE etc."):
        st.code(
            """
# Top 3 ani cu cei mai mari câștiguri:
top3 = mcd.nlargest(3, 'Earnings ($B)')
st.table(top3[['Earnings ($B)']])

# Categorizare ani pe baza 'Net assets ($B)'
q = mcd['Net assets ($B)'].quantile([0.33, 0.67])
def cat(v):
    return ('ani mai puțin buni' if v <= q[0.33]
            else 'ani medii' if v <= q[0.67]
            else 'ani cei mai buni')
mcd['Categorie ani'] = mcd['Net assets ($B)'].map(cat)
st.dataframe(mcd[['Net assets ($B)', 'Categorie ani']])

# Calcul ROE = Earnings / Net assets
roe = (mcd['Earnings ($B)'] / mcd['Net assets ($B)']).dropna()
st.line_chart(roe, height=300)
st.write(roe.to_frame("ROE"))
            """,
            language="python"
        )
    st.markdown("### Top 3 ani profit")
    top3 = mcd.nlargest(3, 'Earnings ($B)')
    st.table(top3[['Earnings ($B)']])
    
    st.markdown("### Categorizare ani după Net assets")
    q = mcd['Net assets ($B)'].quantile([0.33, 0.67])
    def cat(v):
        return ('ani mai puțin buni' if v <= q[0.33]
                else 'ani medii' if v <= q[0.67]
                else 'ani cei mai buni')
    mcd['Categorie ani'] = mcd['Net assets ($B)'].map(cat)
    st.dataframe(mcd[['Net assets ($B)', 'Categorie ani']])
    
    st.markdown("### Calcul ROE")
    roe = (mcd['Earnings ($B)'] / mcd['Net assets ($B)']).dropna()
    st.line_chart(roe, height=300)
    st.write(roe.to_frame("ROE"))
