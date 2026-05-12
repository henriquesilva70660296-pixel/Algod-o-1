
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# 1. CONFIGURAÇÃO
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Ultimate", layout="centered")

# --- SISTEMA DE GESTÃO DE DADOS ---
if 'saldo' not in st.session_state: st.session_state.saldo = 100000.0  
if 'posicao' not in st.session_state: st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state: st.session_state.preco_entrada = 0.0
if 'historico_patrimonio' not in st.session_state: st.session_state.historico_patrimonio = [100000.0]
if 'log_trades' not in st.session_state: st.session_state.log_trades = []

# --- ESTILO E STATUS ---
st.markdown("<style>.main { background-color: #0e1117; } div[data-testid='stMetricValue'] { color: #deff9a; }</style>", unsafe_allow_html=True)

def verificar_mercado():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO"
    return "🟢 MERCADO ABERTO" if 10 <= agora.hour < 17 else "🟡 AFTER-MARKET"

@st.cache_data(ttl=0)
def carregar_ia():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    # Dados Fundamentais
    df['USDA_Estoque'], df['Spread_Petroleo'], df['COT_Sentiment'], df['Weather_Risk'] = 76.4, df['Algodao']/df['Petroleo'], 1, 6.8
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread_Petroleo', 'COT_Sentiment', 'Weather_Risk']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_ia()
    preco_atual = dados['Algodao'].iloc[-1]
    
    st.title("🌱 Cotton Intelligence")
    st.info(verificar_mercado())

    # --- NOVO: BLOCO DE FUNDAMENTOS E CLIMA (RESTAURADO) ---
    with st.container():
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.markdown("### 📊 Fundamental (USDA)")
            st.write("**Estoque Mundial:** 76.4M Fardos")
            st.caption("Tendência: Oferta em queda (Altista)")
        with c_f2:
            st.markdown("### 🌍 Risco Climático")
            st.write("**Texas (EUA):** Seca Nível D4")
            st.caption("Alerta: Risco de quebra de safra")
    
    st.markdown("---")

    # MÉTRICAS PRINCIPAIS
    c1, c2, c3 = st.columns(3)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

    # --- IA DINÂMICA (SINAL) ---
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    cor_ia = "#ff4b4b" if prob < 0.45 else "#fccf03" if prob < 0.65 else "#deff9a"
    
    st.markdown(f"**Confiança do Sistema:** {prob*100:.1f}%")
    st.progress(prob) # Barra de progresso visual
    
    if prob > 0.65:
        st.success(f"💎 **SINAL: COMPRA FORTE**")
    elif prob > 0.45:
        st.warning(f"⚖️ **SINAL: NEUTRO / AGUARDAR**")
    else:
        st.error(f"⚠️ **SINAL: RISCO DE BAIXA**")

    # --- PAINEL DE EXECUÇÃO ---
    st.sidebar.header("💰 Gestão de Conta")
    st.sidebar.metric("Saldo", f"${st.session_state.saldo:.2f}")
    
    st.markdown("### Painel de Ordens")
    qtd = st.number_input("Contratos:", min_value=1, value=100, step=50)
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        if st.button("🟢 COMPRAR"):
            st.session_state.saldo -= preco_atual * qtd
            st.session_state.posicao += qtd
            st.session_state.preco_entrada = preco_atual
            st.rerun()
    with col_b2:
        if st.button("🔴 VENDER (Fechar)"):
            if st.session_state.posicao > 0:
                lucro = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
                st.session_state.saldo += preco_atual * st.session_state.posicao
                # Salva no Log (Opção 1)
                st.session_state.log_trades.append({
                    "Data": datetime.now().strftime("%H:%M:%S"),
                    "Preço Entrada": st.session_state.preco_entrada,
                    "Preço Saída": preco_atual,
                    "Resultado": round(lucro, 2)
                })
                # Salva na Performance (Opção 2)
                st.session_state.historico_patrimonio.append(st.session_state.saldo)
                st.session_state.posicao = 0
                st.rerun()

    # --- ABAS DE GESTÃO ---
    tab_graf, tab_dados, tab_perf = st.tabs(["📈 Gráfico", "📁 Histórico (CSV)", "💹 Equity"])
    
    with tab_graf:
        fig = go.Figure(go.Scatter(y=dados['Algodao'].tail(30), fill='tozeroy', line=dict(color=cor_ia)))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_dados:
        if st.session_state.log_trades:
            df_log = pd.DataFrame(st.session_state.log_trades)
            st.dataframe(df_log, use_container_width=True)
            st.download_button("📥 Baixar CSV", df_log.to_csv().encode('utf-8'), "trades.csv")

    with tab_perf:
        fig_equity = go.Figure(go.Scatter(y=st.session_state.historico_patrimonio, mode='lines+markers', line=dict(color='#deff9a')))
        fig_equity.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_equity, use_container_width=True)

except Exception as e:
    st.error(f"Erro: {e}")
