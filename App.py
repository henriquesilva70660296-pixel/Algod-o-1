import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. ATUALIZAÇÃO INSTANTÂNEA (30 segundos)
st_autorefresh(interval=30 * 1000, key="datarefresh")

st.set_page_config(page_title="Cotton Intel | Premium", layout="centered")

# CSS para customizar a aparência no telemóvel
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #deff9a; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #161b22; 
        border-radius: 5px; 
        padding: 5px 15px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=0)
def carregar_dados_completos():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {}
    for nome, t in tickers.items():
        try:
            df_temp = yf.Ticker(t).history(period="2y")
            if not df_temp.empty: dfs[nome] = df_temp['Close']
        except: continue
    if not dfs: return pd.DataFrame()
    df_final = pd.DataFrame(dfs).ffill().dropna()
    df_final['SMA_9'] = df_final['Algodao'].rolling(9).mean()
    df_final['SMA_21'] = df_final['Algodao'].rolling(21).mean()
    
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=-12.54&longitude=-55.72&start_date=2024-01-01&end_date=2026-05-01&daily=precipitation_sum&timezone=auto"
    try:
        res = requests.get(url).json()
        df_clima = pd.DataFrame(res['daily']).set_index(pd.to_datetime(res['daily']['time']))
        df_final = df_final.join(df_clima, how='left').ffill()
    except: df_final['precipitation_sum'] = 0
    return df_final.dropna()

@st.cache_data(ttl=0)
def carregar_tendencia_1h():
    try: return yf.Ticker("CT=F").history(period="1d", interval="5m")
    except: return pd.DataFrame()

# --- INTERFACE ---
st.title("🌱 Cotton Intelligence")
st.caption("Terminal de Alta Precisão • Real-time")

try:
    dados = carregar_dados_completos()
    dados_1h = carregar_tendencia_1h()

    if not dados.empty:
        # LÓGICA IA
        features = ['Algodao', 'Petroleo', 'Dolar', 'SMA_9', 'SMA_21', 'precipitation_sum']
        X = dados[features][:-1]
        y = (dados['Algodao'].pct_change().shift(-1) > 0.005).astype(int)[:-1]
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        prob = modelo.predict_proba(dados[features].tail(1))[0][1]

        # --- PAINEL DE MÉTRICAS ---
        c1, c2, c3 = st.columns(3)
        preco_atual = dados['Algodao'].iloc[-1]
        c1.metric("ALGODÃO", f"${preco_atual:.2f}", f"{dados['Algodao'].pct_change().iloc[-1]*100:.2f}%")
        c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
        c3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

        # --- SINAL IA ---
        if prob > 0.65:
            st.markdown(f"<h2 style='text-align: center; color: #deff9a;'>🚀 COMPRA FORTE ({prob*100:.0f}%)</h2>", unsafe_allow_html=True)
        elif prob > 0.45:
            st.markdown(f"<h2 style='text-align: center; color: #64748b;'>📈 NEUTRO ({prob*100:.0f}%)</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center; color: #ff4b4b;'>📉 RISCO DE BAIXA ({prob*100:.0f}%)</h2>", unsafe_allow_html=True)

        # --- SEÇÃO DE GRÁFICOS ---
        tab1, tab2 = st.tabs(["📊 Tendência 1h", "📈 Histórico"])
        
        with tab1:
            if len(dados_1h) > 1:
                df_p = dados_1h.tail(12)
                variacao = df_p['Close'].iloc[-1] - df_p['Close'].iloc[0]
                cor_grafico = "#deff9a" if variacao >= 0 else "#ff4b4b"
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['Close'], fill='tozeroy', 
                                         line=dict(color=cor_grafico, width=3),
                                         name="Preço 5min"))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  yaxis=dict(side="right", gridcolor="#333"), xaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Mercado em espera.")

        with tab2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=dados.index, y=dados['Algodao'], line=dict(color='#1f77b4', width=2)))
            fig_hist.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   yaxis=dict(side="right", gridcolor="#333"))
            st.plotly_chart(fig_hist, use_container_width=True)

except Exception as e:
    st.error(f"Aguardando conexão: {e}")

st.caption("Fontes: Yahoo Finance & Global Climate Nodes")
