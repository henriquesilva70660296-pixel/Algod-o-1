import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. ATUALIZAÇÃO INSTANTÂNEA (30s)
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Algodão IA Instantâneo", layout="centered")

@st.cache_data(ttl=0)
def carregar_dados_diarios():
    df = yf.Ticker("CT=F").history(period="2y")
    if df.empty: return pd.DataFrame()
    df['SMA_9'] = df['Close'].rolling(9).mean()
    df['SMA_21'] = df['Close'].rolling(21).mean()
    
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=-12.54&longitude=-55.72&start_date=2024-01-01&end_date=2026-05-01&daily=precipitation_sum&timezone=auto"
    res = requests.get(url).json()
    df_clima = pd.DataFrame(res['daily']).set_index(pd.to_datetime(res['daily']['time']))
    df.index = df.index.tz_localize(None)
    return df.join(df_clima, how='left').ffill().dropna()

@st.cache_data(ttl=0)
def carregar_tendencia_1h():
    # Puxa dados intraday
    df_1h = yf.Ticker("CT=F").history(period="1d", interval="5m")
    return df_1h

# --- INTERFACE ---
st.title("🌱 Cotton Intelligence")
st.caption("Monitorização Instantânea (30s)")

try:
    dados = carregar_dados_diarios()
    dados_1h = carregar_tendencia_1h()

    if not dados.empty:
        # LÓGICA DA IA
        features = ['Close', 'SMA_9', 'SMA_21', 'precipitation_sum']
        X = dados[features][:-1]
        y = (dados['Close'].pct_change().shift(-1) > 0.005).astype(int)[:-1]
        modelo = RandomForestClassifier(n_estimators=50, random_state=42)
        modelo.fit(X, y)
        prob = modelo.predict_proba(dados[features].tail(1))[0][1]

        # MÉTRICAS PRINCIPAIS
        preco_atual = dados['Close'].iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("Preço Atual", f"US$ {preco_atual:.2f}")
        col2.metric("Confiança IA", f"{prob*100:.1f}%")

        # --- GRÁFICO DE TENDÊNCIA 1 HORA (COM PROTEÇÃO) ---
        st.markdown("---")
        st.subheader("📊 Tendência (Última 1 Hora)")
        
        if len(dados_1h) > 1:
            dados_plot_1h = dados_1h.tail(12)
            preco_inicio_hora = dados_plot_1h['Close'].iloc[0]
            variacao_hora = ((preco_atual - preco_inicio_hora) / preco_inicio_hora) * 100
            cor = "green" if variacao_hora >= 0 else "red"
            
            st.markdown(f"Variação: <span style='color:{cor}; font-size:20px; font-weight:bold;'>{variacao_hora:+.2f}%</span>", unsafe_allow_html=True)

            fig_1h = go.Figure()
            fig_1h.add_trace(go.Scatter(x=dados_plot_1h.index, y=dados_plot_1h['Close'], mode='lines+markers', line=dict(color=cor, width=4)))
            fig_1h.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(side="right", tickformat=".2f"))
            st.plotly_chart(fig_1h, use_container_width=True)
        else:
            st.warning("Mercado fechado ou sem dados intraday no momento.")

        # GRÁFICO HISTÓRICO
        st.subheader("📈 Histórico Diário")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço'))
        fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(side="right"))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.error("Não foi possível carregar os dados do mercado.")

except Exception as e:
    st.error(f"Erro na atualização: {e}")

