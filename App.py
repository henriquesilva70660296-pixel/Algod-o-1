import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# Configuração para o ecrã do telemóvel
st.set_page_config(page_title="Algodão IA", layout="centered")

@st.cache_data(ttl=3600)
def carregar_dados():
    # 1. MOTOR DE DADOS: Preços do Algodão (Yahoo Finance)
    df = yf.Ticker("CT=F").history(period="2y")
    df['SMA_9'] = df['Close'].rolling(9).mean()
    df['SMA_21'] = df['Close'].rolling(21).mean()
    
    # 2. MOTOR DE DADOS: Clima em Sorriso-MT (Open-Meteo)
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=-12.54&longitude=-55.72&start_date=2024-01-01&end_date=2026-05-01&daily=precipitation_sum&timezone=auto"
    res = requests.get(url).json()
    df_clima = pd.DataFrame(res['daily']).set_index(pd.to_datetime(res['daily']['time']))
    
    df.index = df.index.tz_localize(None)
    return df.join(df_clima, how='left').ffill().dropna()

# --- INTERFACE ---
st.title("🌱 Cotton Intelligence")

try:
    dados = carregar_dados()

    # 3. MODELO DE IA (Random Forest)
    features = ['Close', 'SMA_9', 'SMA_21', 'precipitation_sum']
    X = dados[features][:-1]
    y = (dados['Close'].pct_change().shift(-1) > 0.005).astype(int)[:-1]

    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X, y)

    # Predição para hoje/amanhã
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]

    # Exibição no Telemóvel
    col1, col2 = st.columns(2)
    col1.metric("Preço Atual", f"US$ {dados['Close'].iloc[-1]:.2f}")
    col2.metric("Confiança de Alta", f"{prob*100:.1f}%")

    if prob > 0.6:
        st.success("🚀 SINAL: COMPRA FORTE")
    elif prob > 0.45:
        st.info("📈 SINAL: NEUTRO / AGUARDAR")
    else:
        st.error("📉 SINAL: TENDÊNCIA DE BAIXA")

    # Gráfico Simples
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço'))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao processar dados: {e}")

st.caption("Fontes: Yahoo Finance & Open-Meteo")
