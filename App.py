import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import sqlite3
import pytz
import feedparser

# --- 1. CONFIGURAÇÃO E ESTABILIDADE ---
st_autorefresh(interval=60 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel Pro", layout="wide")

def init_db():
    conn = sqlite3.connect('cotton_intel.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS conta (id INTEGER PRIMARY KEY, saldo REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, data TEXT, tipo TEXT, entrada REAL, saida REAL, lucro REAL)')
    c.execute('SELECT saldo FROM conta WHERE id = 1')
    if not c.fetchone():
        c.execute('INSERT INTO conta (id, saldo) VALUES (1, 100000.0)')
    conn.commit()
    conn.close()

init_db()

@st.cache_data(ttl=600)
def carregar_dados_mestre():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['Volatilidade'] = df['Algodao'].diff().abs().rolling(14).mean()
    df_norm = (df / df.iloc[0]) * 100
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, df_norm, features

# --- 2. LÓGICA DE HORÁRIO ---
def get_market_status():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    abertura = agora.replace(hour=10, minute=0, second=0)
    fechamento = agora.replace(hour=17, minute=0, second=0)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO", "Abre Segunda às 10h", "#4a1010"
    if agora < abertura: return "🟡 PRÉ-MERCADO", f"Abre às 10h", "#4a4110"
    elif agora > fechamento: return "🔴 MERCADO FECHADO", "Abre Amanhã às 10h", "#4a1010"
    else: return "🟢 MERCADO ABERTO", f"Fecha às 17h", "#104a10"

# --- 3. ESTILO CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric { background-color: #1c2128; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    .status-card { padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 15px; color: white; }
    .ia-container { padding: 20px; border-radius: 15px; text-align: center; border: 2px solid; margin-bottom: 10px; background-color: rgba(0,0,0,0.2); }
    .trading-box { background-color: #1c2128; padding: 20px; border-radius: 15px; border: 1px solid #30363d; height: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. EXECUÇÃO ---
try:
    modelo, df, df_norm, features = carregar_dados_mestre()
    preco_atual = df['Algodao'].iloc[-1]
    volat = df['Volatilidade'].iloc[-1]
    saldo_atual = sqlite3.connect('cotton_intel.db').execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🛡️ Gestão de Risco")
        st.metric("Saldo", f"${saldo_atual:,.2f}")
        risco_p = st.slider("Risco por Operação %", 0.5, 5.0, 2.0)
        lote = int((saldo_atual * (risco_p/100)) / ((volat * 2) * 100)) if volat > 0 else 100
        st.write(f"Lote Sugerido: **{lote} Ct**")
        st.markdown("---")
        st.write("Configurações do Terminal")

    # --- TOPO: STATUS ---
    status_label, tempo_label, cor_fundo = get_market_status()
    st.markdown(f'<div class="status-card" style="background-color: {cor_fundo};"><b>{status_label}</b> | {tempo_label}</div>', unsafe_allow_html=True)

    # --- MÉTRICAS RÁPIDAS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("ALGODÃO", f"${preco_atual:.4f}")
    m2.metric("PETRÓLEO", f"${df['Petroleo'].iloc[-1]:.2f}")
    m3.metric("DÓLAR", f"{df['Dolar'].iloc[-1]:.2f}")

    st.markdown("---")

    # --- PAINEL INICIAL: IA + OPERAÇÃO ---
    col_ia, col_trade = st.columns([1.5, 1])

    with col_ia:
        prob = modelo.predict_proba(df[features].tail(1))[0][1]
        if prob > 0.65: cor_ia, txt_ia = "#deff9a", "TENDÊNCIA DE ALTA"
        elif prob < 0.35: cor_ia, txt_ia = "#ff4b4b", "TENDÊNCIA DE BAIXA"
        else: cor_ia, txt_ia = "#fccf03", "NEUTRO / INDECISO"

        st.markdown(f"""
            <div class="ia-container" style="border-color: {cor_ia}; color: {cor_ia};">
                <small style="color: white; opacity: 0.6;">CONFIANÇA DA IA</small><br>
                <span style="font-size: 50px; font-weight: 900;">{prob*100:.1f}%</span><br>
                <b style="font-size: 20px;">{txt_ia}</b>
            </div>
            """, unsafe_allow_html=True)

    with col_trade:
        st.markdown('<div class="trading-box">', unsafe_allow_html=True)
        qtd = st.number_input("Quantidade (Ct):", 1, 5000, lote)
        
        if 'ent' not in st.session_state:
            if st.button("🟢 EXECUTAR COMPRA", use_container_width=True):
                st.session_state.ent = preco_atual
                st.session_state.q = qtd
                st.rerun()
        else:
            lucro_vivo = (preco_atual - st.session_state.ent) * st.session_state.q
            st.metric("Lucro/Prejuízo Aberto", f"${lucro_vivo:,.2f}", delta=f"{((preco_atual/st.session_state.ent)-1)*100:.2f}%")
            if st.button("🔴 FECHAR POSIÇÃO", use_container_width=True):
                conn = sqlite3.connect('cotton_intel.db')
                conn.execute('UPDATE conta SET saldo = saldo + ?', (lucro_vivo,))
                conn.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro) VALUES (?,?,?,?,?)',
                             (datetime.now().strftime("%d/%m %H:%M"), "LONG", st.session_state.ent, preco_atual, lucro_vivo))
                conn.commit()
                conn.close()
                del st.session_state.ent
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ABAS DE ANÁLISE ---
    tab_g, tab_f, tab_c, tab_n = st.tabs(["📊 Gráfico", "📦 Fundamentos", "🔗 Correlação", "📰 Radar"])

    with tab_g:
        fig = go.Figure(go.Scatter(y=df['Algodao'].tail(60), line=dict(color=cor_ia, width=3), fill='tozeroy'))
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_f:
        f1, f2 = st.columns(2)
        f1.markdown('<div class="stMetric"><b>ESTOQUE USDA</b><br>76.4M Fardos</div>', unsafe_allow_html=True)
        f2.markdown('<div class="stMetric"><b>CLIMA (TEXAS)</b><br>Seca Severa (D4)</div>', unsafe_allow_html=True)

    with tab_c:
        fig_c = go.Figure()
        for c in df_norm.columns: fig_c.add_trace(go.Scatter(y=df_norm[c], name=c))
        fig_c.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_c, use_container_width=True)

    with tab_n:
        feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
        for n in feed.entries[:5]: st.write(f"• {n.title}")

except Exception as e:
    st.info("Sincronizando dados...")

