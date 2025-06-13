import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm

st.set_page_config(page_title="Analiză McDonald's", layout="wide")

# Încarcă și impută valorile lipsă
@st.cache_data
def load_data():
    mcd = pd.read_csv("McDonalds_financial_statements.csv", index_col=0)
    # Înlocuirea tuturor valorilor lipsă
    def nan_replace(tabel):
        assert isinstance(tabel, pd.DataFrame)
        for var in tabel.columns:
            if tabel[var].isna().any():
                if is_numeric_dtype(tabel[var]):
                    tabel[var].fillna(tabel[var].mean(), inplace=True)
                else:
                    tabel[var].fillna(tabel[var].mode()[0], inplace=True)
    nan_replace(mcd)
    # Asigurăm indexul numeric
    mcd.index = mcd.index.astype(int)
    return mcd

@st.cache_data
def load_gdp():
    return pd.read_csv("gdp.csv")

mcd = load_data()
gdp = load_gdp()

st.sidebar.title("Selectează problema")
problema = st.sidebar.radio("", [
    "0. Tratare valori lipsă",
    "1. Raport Datorii/Active",
    "2. Creștere Venituri",
    "3. Marjă vs Datorie",
    "4. Clusterizare",
    "5. Top 3 ani profit",
    "6. Categorizare ani",
    "7. Calcul ROE",
    "8. Regresie multiplă",
    "9. Primul MarketCap>100B",
    "10. Medie profit anual",
    "11. Profit/GDP",
    "12. Ștergere coloane"
])

st.header(problema)

if problema == "0. Tratare valori lipsă":
    st.write("Număr valori lipsă per coloană după imputare:")
    st.dataframe(mcd.isna().sum().to_frame("missing"))

elif problema == "1. Raport Datorii/Active":
    # Problema 1 - calculul raportului datoriilor la active
    ratios = []
    for year, row in mcd.iterrows():
        td = row['Total debt ($B)']
        ta = row['Total assets ($B)']
        if ta != 0:
            ratios.append((year, td / ta))
    df_ratio = pd.Series({y: r for y, r in ratios})
    min_year = df_ratio.idxmin()
    st.line_chart(df_ratio, height=300)
    st.write(f"Anul cu cel mai mic raport Debt/Assets: **{min_year}**")

elif problema == "2. Creștere Venituri":
    # Problema 2 - creștere procentuală anuală a veniturilor
    rev = mcd['Revenue ($B)']
    growth = rev.pct_change() * 100
    max_year = growth.idxmax()
    st.bar_chart(growth.dropna(), height=300)
    st.write(f"Anul cu cea mai mare creștere procentuală: **{max_year}** ({growth.max():.2f}%)")

elif problema == "3. Marjă vs Datorie":
    # Problema 3 - grafic marjă operațională vs datorie
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mcd.index, mcd['Operating Margin (%)'], 'o-', color='blue', label='Marjă Oper.')
    ax2 = ax.twinx()
    ax2.plot(mcd.index, mcd['Total debt ($B)'], 's--', color='red', label='Datorie')
    ax.set_xlabel("An")
    ax.set_ylabel("Marjă (%)")
    ax2.set_ylabel("Datorie ($B)")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)

elif problema == "4. Clusterizare":
    # Problema 4 - clusterizare KMeans pe anii McDonald's
    features = ['Market cap ($B)', 'Revenue ($B)', 'Earnings ($B)',
                'P/E ratio', 'Operating Margin (%)', 'Net assets ($B)',
                'Total assets ($B)', 'Total debt ($B)']
    data = (mcd[features] - mcd[features].mean()) / mcd[features].std()
    k = st.slider("Număr de clustere", 2, 6, 3)
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

elif problema == "5. Top 3 ani profit":
    # Problema 5 - top 3 ani cu cele mai mari câștiguri
    top3 = mcd.nlargest(3, 'Earnings ($B)')
    st.table(top3[['Earnings ($B)']])

elif problema == "6. Categorizare ani":
    # Problema 6 - categorii pe baza cuantilelor Net assets
    q = mcd['Net assets ($B)'].quantile([0.33, 0.67])
    def cat(v):
        return ('ani mai puțin buni' if v <= q[0.33]
                else 'ani medii' if v <= q[0.67]
                else 'ani cei mai buni')
    mcd['Categorie ani'] = mcd['Net assets ($B)'].map(cat)
    st.dataframe(mcd[['Net assets ($B)', 'Categorie ani']])

elif problema == "7. Calcul ROE":
    # Problema 7 - calcul ROE = Earnings / Net assets
    roe = (mcd['Earnings ($B)'] / mcd['Net assets ($B)']).dropna()
    st.line_chart(roe, height=300)
    st.write(roe.to_frame("ROE"))

elif problema == "8. Regresie multiplă":
    # Problema 8 - regresie multiplă Revenue ~ MarketCap + Earnings + Debt
    X = mcd[['Market cap ($B)', 'Earnings ($B)', 'Total debt ($B)']]
    X = sm.add_constant(X)
    y = mcd['Revenue ($B)']
    model = sm.OLS(y, X, missing='drop').fit()
    st.text(model.summary().as_text())

elif problema == "9. Primul MarketCap>100B":
    # Problema 9 - primul an cu market cap > 100
    over = mcd[mcd['Market cap ($B)'] > 100]
    if not over.empty:
        first_year = over.index.min()
        st.write(f"Primul an cu Market cap > 100B: {first_year}")
    else:
        st.write("Niciun an cu Market cap > 100B")


elif problema == "10. Medie profit anual":
    # Problema 10 - media câștigurilor pe an
    avg = mcd['Earnings ($B)'].groupby(mcd.index).mean()
    mcd['Average Earnings ($B)'] = avg
    st.line_chart(avg, height=300)

elif problema == "11. Profit/GDP":
    # Problema 11 - combin area cu PIB și calcul Profit per GDP
    merged = mcd.reset_index().merge(gdp, on="Year")
    merged['Profit per GDP'] = merged['Earnings ($B)'] / merged['GDP']
    st.line_chart(merged.set_index('Year')['Profit per GDP'], height=300)
    corr = merged['Earnings ($B)'].corr(merged['GDP'])
    st.write(f"Coeficient de corelație: {corr:.2f}")

elif problema == "12. Ștergere coloane":
    # Problema 12 - eliminarea coloanelor P/S și P/B și filtrarea Revenue>=20
    df2 = mcd.drop(columns=['P/S ratio', 'P/B ratio'])
    df2 = df2[df2['Revenue ($B)'] >= 20]
    st.dataframe(df2)
