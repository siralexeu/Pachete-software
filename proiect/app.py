import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm

# Configurare pagină și includere CSS personalizat pentru stilizare
st.set_page_config(
    page_title="Analiză McDonald's",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stilizare CSS personalizată
st.markdown(
    """
    <style>
        .main-header {
            padding: 20px 0;
            text-align: center;
            border-bottom: 2px solid #1E3050;
            margin-bottom: 30px;
        }
        .sidebar-header {
            font-size: 24px;
            color: #1E3050;
            padding: 10px 0;
        }
        .stRadio > div {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        /* Am eliminat fundalul alb, folosind transparent */
        .chart-container {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Încărcare și imputare valori lipsă
@st.cache_data
def load_data():
    mcd = pd.read_csv("proiect/McDonalds_financial_statements.csv", index_col=0)
    
    # Înlocuirea valorilor lipsă
    def nan_replace(tabel):
        assert isinstance(tabel, pd.DataFrame)
        for var in tabel.columns:
            if tabel[var].isna().any():
                if is_numeric_dtype(tabel[var]):
                    tabel[var].fillna(tabel[var].mean(), inplace=True)
                else:
                    tabel[var].fillna(tabel[var].mode()[0], inplace=True)
    
    nan_replace(mcd)
    # Asigurăm că indexul este numeric
    mcd.index = mcd.index.astype(int)
    return mcd

@st.cache_data
def load_gdp():
    return pd.read_csv("proiect/gdp.csv")

mcd = load_data()
gdp = load_gdp()

# Sidebar îmbunătățit cu iconițe și stilizat
with st.sidebar:
    st.markdown('<p class="sidebar-header">Meniu Analiză</p>', unsafe_allow_html=True)
    optiuni = [
        "📊 Tratare valori lipsă",
        "💰 Raport Datorii/Active",
        "📈 Creștere Venituri",
        "📉 Marjă vs Datorie",
        "🔍 Clusterizare",
        "🏆 Top 3 ani profit",
        "📋 Categorizare ani",
        "📊 Calcul ROE",
        "📈 Regresie multiplă",
        "💹 Primul MarketCap>100B",
        "💵 Medie profit anual",
        "🌍 Profit/GDP",
        "🗑️ Ștergere coloane"
    ]
    problema = st.radio("Selectează secțiunea:", optiuni)

# Header principal
st.markdown(
    '<h1 class="main-header">Analiză Financiară McDonald\'s</h1>',
    unsafe_allow_html=True
)

# Container pentru conținut
with st.container():
    # Extrage titlul secțiunii fără iconiță
    subtitlu = problema.split(" ", 1)[1]
    st.subheader(subtitlu)
    st.markdown("---")
    
    # Wrapper pentru conținut (grafic, tabele etc.)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Mapare între opțiunile noi și etichetele folosite în logica inițială
    problema_map = {
        "📊 Tratare valori lipsă": "0. Tratare valori lipsă",
        "💰 Raport Datorii/Active": "1. Raport Datorii/Active",
        "📈 Creștere Venituri": "2. Creștere Venituri",
        "📉 Marjă vs Datorie": "3. Marjă vs Datorie",
        "🔍 Clusterizare": "4. Clusterizare",
        "🏆 Top 3 ani profit": "5. Top 3 ani profit",
        "📋 Categorizare ani": "6. Categorizare ani",
        "📊 Calcul ROE": "7. Calcul ROE",
        "📈 Regresie multiplă": "8. Regresie multiplă",
        "💹 Primul MarketCap>100B": "9. Primul MarketCap>100B",
        "💵 Medie profit anual": "10. Medie profit anual",
        "🌍 Profit/GDP": "11. Profit/GDP",
        "🗑️ Ștergere coloane": "12. Ștergere coloane"
    }
    
    # Folosim key-ul mapat pentru a păstra logica inițială
    current_problem = problema_map.get(problema, problema)
    
    # Afișare conținut în funcție de opțiunea selectată
    if current_problem == "0. Tratare valori lipsă":
        st.write("Număr valori lipsă per coloană după imputare:")
        st.dataframe(mcd.isna().sum().to_frame("missing"))
    
    elif current_problem == "1. Raport Datorii/Active":
        # Calculul raportului datoriilor la active
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
    
    elif current_problem == "2. Creștere Venituri":
        # Creștere procentuală anuală a veniturilor
        rev = mcd['Revenue ($B)']
        growth = rev.pct_change() * 100
        max_year = growth.idxmax()
        st.bar_chart(growth.dropna(), height=300)
        st.write(
            f"Anul cu cea mai mare creștere procentuală: "
            f"**{max_year}** ({growth.max():.2f}%)"
        )
    
    elif current_problem == "3. Marjă vs Datorie":
        # Grafic marjă operațională vs datorie
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            mcd.index, mcd['Operating Margin (%)'], 'o-', color='blue', 
            label='Marjă Oper.'
        )
        ax2 = ax.twinx()
        ax2.plot(
            mcd.index, mcd['Total debt ($B)'], 's--', color='red',
            label='Datorie'
        )
        ax.set_xlabel("An")
        ax.set_ylabel("Marjă (%)")
        ax2.set_ylabel("Datorie ($B)")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        st.pyplot(fig)
    
    elif current_problem == "4. Clusterizare":
        # Clusterizare KMeans pe anii McDonald's
        features = [
            'Market cap ($B)', 'Revenue ($B)', 'Earnings ($B)',
            'P/E ratio', 'Operating Margin (%)', 'Net assets ($B)',
            'Total assets ($B)', 'Total debt ($B)'
        ]
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
    
    elif current_problem == "5. Top 3 ani profit":
        # Top 3 ani cu cele mai mari câștiguri
        top3 = mcd.nlargest(3, 'Earnings ($B)')
        st.table(top3[['Earnings ($B)']])
    
    elif current_problem == "6. Categorizare ani":
        # Categorii pe baza cuantilelor Net assets
        q = mcd['Net assets ($B)'].quantile([0.33, 0.67])
    
        def cat(v):
            return (
                'ani mai puțin buni'
                if v <= q[0.33]
                else 'ani medii'
                if v <= q[0.67]
                else 'ani cei mai buni'
            )
    
        mcd['Categorie ani'] = mcd['Net assets ($B)'].map(cat)
        st.dataframe(mcd[['Net assets ($B)', 'Categorie ani']])
    
    elif current_problem == "7. Calcul ROE":
        # Calcul ROE = Earnings / Net assets
        roe = (mcd['Earnings ($B)'] / mcd['Net assets ($B)']).dropna()
        st.line_chart(roe, height=300)
        st.write(roe.to_frame("ROE"))
    
    elif current_problem == "8. Regresie multiplă":
        # Regresie multiplă: Revenue ~ Market cap + Earnings + Total debt
        X = mcd[['Market cap ($B)', 'Earnings ($B)', 'Total debt ($B)']]
        X = sm.add_constant(X)
        y = mcd['Revenue ($B)']
        model = sm.OLS(y, X, missing='drop').fit()
        st.text(model.summary().as_text())
    
    elif current_problem == "9. Primul MarketCap>100B":
        # Primul an cu Market cap > 100B
        over = mcd[mcd['Market cap ($B)'] > 100]
        if not over.empty:
            first_year = over.index.min()
            st.write(f"Primul an cu Market cap > 100B: {first_year}")
        else:
            st.write("Niciun an cu Market cap > 100B")
    
    elif current_problem == "10. Medie profit anual":
        # Media câștigurilor pe an
        avg = mcd['Earnings ($B)'].groupby(mcd.index).mean()
        mcd['Average Earnings ($B)'] = avg
        st.line_chart(avg, height=300)
    
    elif current_problem == "11. Profit/GDP":
        # Combinarea datelor cu PIB și calcularea raportului Profit per GDP
        merged = mcd.reset_index().merge(gdp, on="Year")
        merged['Profit per GDP'] = merged['Earnings ($B)'] / merged['GDP']
        st.line_chart(merged.set_index('Year')['Profit per GDP'], height=300)
        corr = merged['Earnings ($B)'].corr(merged['GDP'])
        st.write(f"Coeficient de corelație: {corr:.2f}")
    
    elif current_problem == "12. Ștergere coloane":
        # Ștergerea coloanelor P/S ratio și P/B ratio și filtrarea rândurilor cu Revenue >= 20
        df2 = mcd.drop(columns=['P/S ratio', 'P/B ratio'])
        df2 = df2[df2['Revenue ($B)'] >= 20]
        st.dataframe(df2)
    
    # Închidem containerul wrapper
    st.markdown("</div>", unsafe_allow_html=True)
