import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import sqlite3
from twilio.rest import Client
import pytz

# 1. CONFIGURAÇÃO E BANCO DE DADOS
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | WhatsApp Alert", layout="centered")

# --- SISTEMA DE NOTIFICAÇÃO (WHATSAPP VIA TWILIO) ---
def enviar_alerta_whatsapp(mensagem):
    # Credenciais inseridas na barra lateral por segurança
    account_sid = st.sidebar.text_input("Twilio Account SID", type="password")
    auth_token = st.sidebar.text_input("Twilio Auth Token", type="password")
    seu_numero = st.sidebar.text_input("Seu Número (ex: +55...)", placeholder="+5511999999999")
    
    if account_sid and auth_token and seu_numero:
        try:
            client = Client(account_sid, auth_token)
            # O número da Twilio para o Sandbox geralmente é +14155238886
            message = client.messages.create(
                from_='whatsapp:+14155238886', 
                body=mensagem,
                to=f'whatsapp:{seu_numero}'
            )
            return True
        except Exception as e:
            st.sidebar.error(f"Erro WhatsApp: {e}")
            return False
    return False

# --- INICIALIZAÇÃO DB ---
def init_db():
    conn = sqlite3.connect('cotton_intel.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS conta (id INTEGER PRIMARY KEY, saldo REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, data TEXT, tipo TEXT, entrada REAL, saida REAL, lucro REAL)')
    c.execute('SELECT saldo FROM conta WHERE id = 1')
    if not c.fetchone(): c.execute('INSERT INTO conta (id, saldo) VALUES (1, 100000.0)')
    conn.commit()
    conn.close()

init_db()

# --- ESTILO E MERCADO ---
st.markdown("<style>.main { background-color: #0e1117; } .status-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def carregar_dados():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    df['USDA_Estoque'], df['Spread'], df['COT'], df['Weather'] = 76.4, df['Algodao']/df['Petroleo'], 1, 6.8
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread', 'COT', 'Weather']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_dados()
    preco_atual = dados['Algodao'].iloc[-1]
    
    # STATUS E HEADER
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    status_m = "🟢 MERCADO ABERTO" if 10 <= agora.hour < 17 and agora.weekday() < 5 else "🔴 MERCADO FECHADO"
    st.markdown(f'<div class="status-box">{status_m}</div>', unsafe_allow_html=True)
    st.title("🌱 Cotton Intelligence + WhatsApp")

    # IA E SINAL
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    cor_sinal = "#deff9a" if prob > 0.65 else "#ff4b4b" if prob < 0.35 else "#fccf03"
    
    st.markdown(f"**Confiança da IA:** <span style='color:{cor_sinal}'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
    st.progress(prob)

    # --- LÓGICA DE ALERTA WHATSAPP ---
    if prob > 0.85:
        msg = f"🚀 ALERTA COTTON: Compra Forte! Preço: ${preco_atual:.2f} | Confiança: {prob*100:.1f}%"
        if st.sidebar.button("📲 Testar Envio WhatsApp"):
            if enviar_alerta_whatsapp(msg):
                st.sidebar.success("Mensagem enviada!")

    # GESTÃO (SALDO DB)
    conn = sqlite3.connect('cotton_intel.db')
    saldo_db = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    st.sidebar.metric("Saldo Permanente", f"${saldo_db:.2f}")

    # PAINEL DE ORDENS
    qtd = st.number_input("Contratos:", 1, 1000, 100)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🟢 COMPRAR", use_container_width=True):
            st.session_state.p_entrada = preco_atual
            st.session_state.p_qtd = qtd
            st.success("Ordem Executada")
    with col_b2:
        if st.button("🔴 FECHAR POSIÇÃO", use_container_width=True):
            if 'p_entrada' in st.session_state:
                lucro = (preco_atual - st.session_state.p_entrada) * st.session_state.p_qtd
                conn.execute('UPDATE conta SET saldo = ? WHERE id = 1', (saldo_db + lucro,))
                conn.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro) VALUES (?,?,?,?,?)', 
                             (datetime.now().strftime("%d/%m %H:%M"), "LONG", st.session_state.p_entrada, preco_atual, lucro))
                conn.commit()
                del st.session_state.p_entrada
                st.rerun()
    conn.close()

    # GRÁFICO E HISTÓRICO
    t1, t2 = st.tabs(["📊 Gráfico Técnico", "📁 Histórico Salvo"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dados['Algodao'].tail(50), name="Preço", line=dict(color=cor_sinal, width=3)))
        fig.add_trace(go.Scatter(y=dados['MA20'].tail(50), name="Média 20", line=dict(color='rgba(255,255,255,0.3)', dash='dot')))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        conn = sqlite3.connect('cotton_intel.db')
        st.dataframe(pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn), use_container_width=True)
        conn.close()

except Exception as e:
    st.error(f"Erro no Sistema: {e}")

