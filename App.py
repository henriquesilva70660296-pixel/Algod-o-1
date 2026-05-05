import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. ATUALIZAÇÃO INSTANTÂNEA
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Terminal Pro", layout="centered")

# --- SISTEMA DE CARTEIRA ATUALIZADO ---
if 'saldo' not in st.session_state:
    st.session_state.saldo = 100000.0  # Saldo aumentado para evitar erro de margem
if 'posicao' not in st.session_state:
    st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state:
    st.session_state.preco_entrada = 0.0

@st.cache_data(ttl=0)
def carregar_dados():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['SMA_9'] = df['Algodao'].rolling(9).mean()
    df['SMA_21'] = df['Algodao'].rolling(21).mean()
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    return df.dropna()

# --- INTERFACE ---
st.title("🚀 Terminal de Trading Algodão")

try:
    dados = carregar_dados()
    preco_atual = dados['Algodao'].iloc[-1]
    
    # INTELIGÊNCIA ARTIFICIAL
    features = ['Algodao', 'Petroleo', 'Dolar', 'SMA_9', 'SMA_21']
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(dados[features][:-1], dados['Target'][:-1])
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]

    # --- BARRA LATERAL (GESTÃO DE CONTA) ---
    st.sidebar.header("💰 Minha Conta")
    st.sidebar.metric("Saldo Disponível", f"US$ {st.session_state.saldo:.2f}")
    
    if st.session_state.posicao > 0:
        pnl = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
        st.sidebar.metric("Lucro/Prejuízo Aberto", f"US$ {pnl:.2f}", delta=f"{pnl:.2f}")
        st.sidebar.write(f"Contratos Ativos: {st.session_state.posicao}")

    # --- PAINEL DE EXECUÇÃO ---
    st.subheader("Painel de Execução")
    
    # NOVO: Campo de texto para escolher a quantidade
    qtd_operar = st.number_input("Quantidade de Contratos (un):", min_value=1, max_value=10000, value=100, step=10)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("🟢 COMPRAR"):
            custo = preco_atual * qtd_operar
            if st.session_state.saldo >= custo:
                st.session_state.saldo -= custo
                st.session_state.posicao += qtd_operar
                st.session_state.preco_entrada = preco_atual
                st.success(f"Comprou {qtd_operar} unidades!")
            else:
                st.error("Saldo Insuficiente para esta quantidade!")

    with c2:
        if st.button("🔴 VENDER (Zerar)"):
            if st.session_state.posicao > 0:
                receita = preco_atual * st.session_state.posicao
                st.session_state.saldo += receita
                st.session_state.posicao = 0
                st.session_state.preco_entrada = 0
                st.warning("Posição Encerrada com Sucesso!")
            else:
                st.info("Sem posições abertas.")

    with c3:
        st.metric("Preço de Entrada", f"${st.session_state.preco_entrada:.2f}")

    st.markdown("---")
    
    # ANÁLISE E GRÁFICO
    st.subheader("Análise da IA")
    if prob > 0.6:
        st.success(f"SINAL: COMPRA FORTE ({prob*100:.0f}%)")
    else:
        st.info(f"SINAL: AGUARDAR ({prob*100:.0f}%)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Algodao'], name="Preço", line=dict(color='#deff9a')))
    if st.session_state.posicao > 0:
        fig.add_hline(y=st.session_state.preco_entrada, line_dash="dash", line_color="white", annotation_text="Preço de Entrada")
    
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro no terminal: {e}")

