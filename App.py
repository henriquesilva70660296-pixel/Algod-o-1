import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ====================================================================
# [ADICIONADO] MOTOR DE ATUALIZAÇÃO AUTOMÁTICA MINUTO A MINUTO
# ====================================================================
from streamlit_autorefresh import st_autorefresh

# Força o aplicativo a rodar o script sozinho a cada 60 segundos (60000ms)
st_autorefresh(interval=60000, limit=1000, key="cotton_auto_refresh")
# ====================================================================


# ====================================================================
# CONFIGURAÇÃO DA TELA DO APLICATIVO
# ====================================================================
st.set_page_config(page_title="Cotton Intelligence", layout="wide")
st.title("🌾 Cotton Intelligence - Real-Time Dashboard")
st.write("Monitoramento em tempo real e análise de tendências do ativo do Algodão.")


# ====================================================================
# BUSCA DE DADOS DO ALGODÃO (CONTRATOS FUTUROS DA BOLSA DE NY)
# ====================================================================
# Usando intervalo de 1 minuto ('1m') para capturar as movimentações do dia
ticker_algodao = "CT=F"

try:
    dados = yf.download(tickers=ticker_algodao, period="1d", interval="1m")
    
    if not dados.empty:
        # Puxando o último preço disponível da API
        preco_atual = dados['Close'].iloc[-1]
        variacao = preco_atual - dados['Open'].iloc[0]
        variacao_pct = (variacao / dados['Open'].iloc[0]) * 100

        # Exibição dos cards de preço no topo
        col1, col2 = st.columns(2)
        col1.metric(label="Preço Atual do Algodão (Bolsa de NY)", value=f"US$ {preco_atual:.4f}")
        col2.metric(label="Variação Diária", value=f"{variacao:.4f}", delta=f"{variacao_pct:.2f}%")

        # ====================================================================
        # CÁLCULO DOS INDICADORES TÉCNICOS
        # ====================================================================
        # Média Móvel de 20 períodos (MA20)
        dados['MA20'] = dados['Close'].rolling(window=20).mean()

        # Índice de Força Relativa (RSI 14)
        delta = dados['Close'].diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = ganho / perda
        dados['RSI'] = 100 - (100 / (1 + rs))


        # ====================================================================
        # CONSTRUÇÃO DO GRÁFICO TÉCNICO INTERATIVO
        # ====================================================================
        fig = go.Figure()

        # Linha principal do preço de fechamento
        fig.add_trace(go.Scatter(
            x=dados.index, 
            y=dados['Close'], 
            mode='lines', 
            name='Preço (1m)', 
            line=dict(color='#00CF85', width=2)
        ))

        # Linha da Média Móvel MA20
        fig.add_trace(go.Scatter(
            x=dados.index, 
            y=dados['MA20'], 
            mode='lines', 
            name='Média Móvel (MA20)', 
            line=dict(color='#FFA500', width=1.5, dash='dash')
        ))

        fig.update_layout(
            title="Gráfico Técnico de Curto Prazo (Intervalo de 1 Minuto)",
            xaxis_title="Horário",
            yaxis_title="Preço (US$)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)


        # ====================================================================
        # PAINEL DO RSI (MOMENTUM DO MERCADO)
        # ====================================================================
        st.subheader("📊 Indicador de Tendência - RSI (14)")
        rsi_atual = dados['RSI'].iloc[-1]
        
        if pd.isna(rsi_atual):
            st.info("Aguardando acumulação de dados para calcular o RSI...")
        else:
            st.write(f"O RSI atual é de **{rsi_atual:.2f}**")
            if rsi_atual > 70:
                st.error("⚠️ Alerta: Ativo em região de Sobrecompra (Pode indicar correção para queda).")
            elif rsi_atual < 30:
                st.success("✅ Alerta: Ativo em região de Sobrevenda (Pode indicar oportunidade de subida).")
            else:
                st.warning("⚖️ Mercado Neutro: Tendência lateralizada no momento.")

    else:
        st.warning("⚠️ O mercado de Algodão está fechado ou sem dados no momento. Verifique durante o horário da Bolsa de NY.")

except Exception as e:
    st.error(f"Erro ao conectar com a API de dados: {e}")
