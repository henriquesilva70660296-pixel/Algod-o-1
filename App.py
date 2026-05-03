import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# Configuração para visualização em telemóvel
st.set_page_config(page_title="Algodão IA", layout="centered")

@st.cache_data(ttl=3600)
def carregar_dados():
    # 1. Obter dados do mercado (Yahoo Finance)
    # Ticker CT=F corresponde ao Futuros de Algodão No. 2
    df = yf.Ticker("CT=F").history(period="2y")
    
    # Indicadores Técnicos Básicos
    df['SMA_9'] = df['Close'].rolling(9).mean()
    df['SMA_21'] = df['Close'].rolling(21).mean()
    
    # 2. Obter dados climáticos (Open-Meteo)
    # Coordenadas aproximadas de Sorriso - MT
    url_clima = "https://archive-api.open-meteo.com/v1/archive?latitude=-12.54&longitude=-55.72&start_date=2024-01-01&end_date=2026-05-01&daily=precipitation_sum&timezone=auto"
    res = requests.get(url_clima).json()
    df_clima = pd.DataFrame(res['daily']).set_index(pd.to_datetime(res['daily']['time']))
    
    # Ajustar fusos horários e unir tabelas
    df.index = df.index.tz_localize(None)
    return df.join(df_clima, how='left').ffill().dropna()

# --- INTERFACE DO UTILIZADOR ---
st.title("🌱 Cotton Intelligence")
st.markdown("### Sistema de Predição Algodão + Clima")

try:
    dados = carregar_dados()

    # --- LÓGICA DA IA ---
    # Usamos Preço, Médias Móveis e Precipitação para prever a tendência
    features = ['Close', 'SMA_9', 'SMA_21', 'precipitation_sum']
    X = dados[features][:-1]
    # Alvo: O preço sobe mais de 0.5% no dia seguinte?
    y = (dados['Close'].pct_change().shift(-1) > 0.005).astype(int)[:-1]

    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X, y)

    # Predição para o próximo movimento
    ultima_linha = dados[features].tail(1)
    prob = modelo.predict_proba(ultima_linha)[0][1]

    # Exibição de Resultados (Métricas)
    col1, col2 = st.columns(2)
    col1.metric("Preço Atual", f"US$ {dados['Close'].iloc[-1]:.2f}")
    col2.metric("Confiança de Alta", f"{prob*100:.1f}%")

    # Painel de Sinal
    if prob > 0.6:
        st.success("🚀 SINAL: COMPRA FORTE")
    elif prob > 0.45:
        st.info("📈 SINAL: TENDÊNCIA NEUTRA / AGUARDAR")
    else:
        st.error("📉 SINAL: TENDÊNCIA DE BAIXA")

    # Gráfico para Telemóvel
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço'))
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_21'], name='Média 21d', line=dict(dash='dot')))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")

st.caption("Fontes: Yahoo Finance & Open-Meteo. Localização: Mato Grosso, BR.")
