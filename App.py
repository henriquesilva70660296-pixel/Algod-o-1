import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. ATUALIZAÇÃO INSTANTÂNEA (30 segundos)
st_autorefresh(interval=30 * 1000, key="datarefresh")

st.set_page_config(page_title="Cotton Intelligence 2.0", layout="centered")

@st.cache_data(ttl=0)
def carregar_dados_completos():
    # --- BUSCA DE DADOS ---
    # Algodão (CT=F), Petróleo (CL=F) e Dólar (DX-Y.NYB ou USDBRL=X)
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    
    data_fim = pd.Timestamp.now()
    data_ini = data_fim - pd.DateOffset(years=2)
    
    dfs = {}
    for nome, t in tickers.items():
        df_temp = yf.Ticker(t).history(start=data_ini, end=data_fim)
        dfs[nome] = df_temp['Close']

    df_final = pd.DataFrame(dfs).ffill().dropna()
    
    # Médias Móveis para o Algodão
    df_final['SMA_9'] = df_final['Algodao'].rolling(9).mean()
    df_final['SMA_21'] = df_final['Algodao'].rolling(21).mean()
    
    # Dados Climáticos (Sorriso-MT)
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=-12.54&longitude=-55.72&start_date=2024-01-01&end_date=2026-05-01&daily=precipitation_sum&timezone=auto"
    try:
        res = requests.get(url).json()
        df_clima = pd.DataFrame(res['daily']).set_index(pd.to_datetime(res['daily']['time']))
        df_final = df_final.join(df_clima, how='left').ffill()
    except:
        df_final['precipitation_sum'] = 0 # Fallback se a API de clima falhar
        
    return df_final.dropna()

@st.cache_data(ttl=0)
def carregar_tendencia_1h():
    df_1h = yf.Ticker("CT=F").history(period="1d", interval="5m")
    return df_1h

# --- INTERFACE ---
st.title("🌱 Cotton Intelligence 2.0")
st.caption("Monitorização Macro & Instantânea (30s)")

try:
    dados = carregar_dados_completos()
    dados_1h = carregar_tendencia_1h()

    if not dados.empty:
        # 3. INTELIGÊNCIA ARTIFICIAL (Agora com Petróleo e Dólar!)
        features = ['Algodao', 'Petroleo', 'Dolar', 'SMA_9', 'SMA_21', 'precipitation_sum']
        X = dados[features][:-1]
        y = (dados['Algodao'].pct_change().shift(-1) > 0.005).astype(int)[:-1]
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        prob = modelo.predict_proba(dados[features].tail(1))[0][1]

        # MÉTRICAS DASHBOARD
        c1, c2, c3 = st.columns(3)
        c1.metric("Algodão", f"${dados['Algodao'].iloc[-1]:.2f}")
        c2.metric("Petróleo (WTI)", f"${dados['Petroleo'].iloc[-1]:.1f}")
        c3.metric("Dólar (DXY)", f"{dados['Dolar'].iloc[-1]:.1f}")

        st.subheader(f"Confiança de Alta IA: {prob*100:.1f}%")
        if prob > 0.65: st.success("🚀 SINAL: COMPRA FORTE (Macro Favorável)")
        elif prob > 0.45: st.info("📈 SINAL: AGUARDAR")
        else: st.error("📉 SINAL: TENDÊNCIA DE BAIXA / RISCO")

        # --- GRÁFICOS ---
        tab1, tab2 = st.tabs(["Tendência 1h", "Histórico 2 Anos"])
        
        with tab1:
            if len(dados_1h) > 1:
                fig_1h = go.Figure()
                fig_1h.add_trace(go.Scatter(x=dados_1h.index, y=dados_1h['Close'], mode='lines+markers', line=dict(color='#deff9a', width=3)))
                fig_1h.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(side="right"))
                st.plotly_chart(fig_1h, use_container_width=True)
            else:
                st.warning("Mercado fechado no momento.")

        with tab2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=dados.index, y=dados['Algodao'], name='Preço Algodão'))
            fig_hist.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(side="right"))
            st.plotly_chart(fig_hist, use_container_width=True)
            
except Exception as e:
    st.error(f"Erro na sincronização: {e}")
