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
from deep_translator import GoogleTranslator

# --- 1. CONFIGURAÇÃO E ESTABILIDADE ---
st_autorefresh(interval=44 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel Pro MASTER", layout="wide")

def init_db():
    conn = sqlite3.connect('cotton_intel.db')
    c = conn.cursor()
    # Criação das tabelas base[span_0](start_span)[span_0](end_span)
    c.execute('CREATE TABLE IF NOT EXISTS conta (id INTEGER PRIMARY KEY, saldo REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, data TEXT, tipo TEXT, entrada REAL, saida REAL, lucro REAL, confianca REAL)')
    
    # CORREÇÃO AUTOMÁTICA DO ERRO DE COLUNA[span_1](start_span)[span_1](end_span)
    try:
        c.execute('ALTER TABLE trades ADD COLUMN confianca REAL DEFAULT 0.5')
    except:
        pass # Coluna já existe
        
    c.execute('SELECT saldo FROM conta WHERE id = 1')
    if not c.fetchone():
        c.execute('INSERT INTO conta (id, saldo) VALUES (1, 100000.0)')
    conn.commit()
    conn.close()

init_db()

@st.cache_data(ttl=60)
def analisar_sentimento_noticias():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
        score = 0
        p_alta = ['rise', 'high', 'shortage', 'bullish', 'increase', 'drought', 'demand', 'low stocks']
        p_baixa = ['fall', 'low', 'surplus', 'bearish', 'decrease', 'oversupply', 'drop', 'weak demand']
        for n in feed.entries[:10]:
            t = n.title.lower()
            if any(w in t for w in p_alta): score += 1
            if any(w in t for w in p_baixa): score -= 1
        return 1 if score > 0 else -1 if score < 0 else 0
    except:
        return 0

@st.cache_data(ttl=40)
def carregar_dados_mestre():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="2y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # Indicadores Técnicos Master[span_2](start_span)[span_2](end_span)
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    delta = df['Algodao'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Sentimento'] = analisar_sentimento_noticias()
    
    df = df.dropna()
    df['Volatilidade'] = df['Algodao'].diff().abs().rolling(14).mean()
    df_norm = (df / df.iloc[0]) * 100
    df['Target'] = (df['Algodao'].shift(-1) > (df['Algodao'] * 1.0005)).astype(int)
    
    features = ['Algodao', 'Petroleo', 'Dolar', 'MA20', 'RSI', 'Sentimento']
    modelo = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced').fit(df[features][:-1], df['Target'][:-1])
    
    return modelo, df, df_norm, features

def get_market_status():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    abertura, fechamento = agora.replace(hour=10, minute=0, second=0), agora.replace(hour=17, minute=0, second=0)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO", "Abre Segunda", "#4a1010"
    return ("🟢 MERCADO ABERTO", "Fecha às 17h", "#104a10") if abertura <= agora <= fechamento else ("🔴 MERCADO FECHADO", "Abre amanhã", "#4a1010")

# --- 2. ESTILO CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .status-card { padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 15px; color: white; font-weight: bold; }
    .ia-container { padding: 25px; border-radius: 15px; text-align: center; border: 2px solid; margin-bottom: 15px; background: linear-gradient(145deg, #0d1117, #161b22); }
    .news-card { background-color: #161b22; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #58a6ff; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LÓGICA DE INTERFACE ---
try:
    modelo, df, df_norm, features = carregar_dados_mestre()
    prob = modelo.predict_proba(df[features].tail(1))[0][1]
    preco_atual = df['Algodao'].iloc[-1]
    
    conn = sqlite3.connect('cotton_intel.db')
    saldo_atual = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    historico = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
    conn.close()

    with st.sidebar:
        st.header("💳 Financial Hub")
        st.metric("Saldo Líquido", f"${saldo_atual:,.2f}")
        
        if not historico.empty:
            win_rate = (len(historico[historico['lucro'] > 0]) / len(historico)) * 100
            st.metric("Taxa de Acerto IA", f"{win_rate:.1f}%")
        
        with st.expander("🛠️ Dados Técnicos"):
            st.write(f"RSI (14): **{df['RSI'].iloc[-1]:.2f}**")
            st.write(f"Volatilidade: **{df['Volatilidade'].iloc[-1]:.4f}**")
        st.markdown("---")
        st.caption("Cotton Intel MASTER v3.1")

    st_lab, t_lab, color = get_market_status()
    st.markdown(f'<div class="status-card" style="background-color: {color};">{st_lab} | {t_lab}</div>', unsafe_allow_html=True)

    c_ia, c_op = st.columns([1.5, 1])

    with c_ia:
        cor_ia, txt_ia = ("#deff9a", "COMPRA FORTE") if prob > 0.75 else ("#ff4b4b", "VENDA FORTE") if prob < 0.25 else ("#fccf03", "AGUARDAR")
        st.markdown(f'<div class="ia-container" style="border-color: {cor_ia}; color: {cor_ia};"><small style="color:#8b949e">CONFIANÇA DO MODELO</small><br><span style="font-size: 55px; font-weight: 900;">{prob*100:.1f}%</span><br><b style="font-size: 22px;">{txt_ia}</b></div>', unsafe_allow_html=True)

    with c_op:
        st.markdown('<div style="background-color:#161b22; padding:20px; border-radius:15px; border:1px solid #30363d">', unsafe_allow_html=True)
        if 'ent' not in st.session_state:
            qtd = st.number_input("Contratos", 1, 1000, 10)
            if st.button("🚀 ABRIR COMPRA", use_container_width=True):
                st.session_state.ent, st.session_state.q, st.session_state.tipo = preco_atual, qtd, "LONG"
                st.rerun()
            if st.button("📉 ABRIR VENDA", use_container_width=True):
                st.session_state.ent, st.session_state.q, st.session_state.tipo = preco_atual, qtd, "SHORT"
                st.rerun()
        else:
            m = 1 if st.session_state.tipo == "LONG" else -1
            lucro = (preco_atual - st.session_state.ent) * st.session_state.q * m
            st.metric(f"Posição {st.session_state.tipo}", f"${lucro:,.2f}", delta=f"{((preco_atual/st.session_state.ent)-1)*100*m:.3f}%")
            if st.button("✅ FECHAR OPERAÇÃO", use_container_width=True):
                c = sqlite3.connect('cotton_intel.db')
                c.execute('UPDATE conta SET saldo = saldo + ?', (lucro,))
                c.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro, confianca) VALUES (?,?,?,?,?,?)',
                         (datetime.now().strftime("%d/%m %H:%M"), st.session_state.tipo, st.session_state.ent, preco_atual, lucro, prob))
                c.commit(); c.close()
                del st.session_state.ent
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📈 Gráfico Master", "📜 Histórico de Performance", "📰 Notícias Traduzidas"])

    with t1:
        fig = go.Figure(data=[go.Scatter(y=df['Algodao'].tail(60), line=dict(color=cor_ia, width=3), fill='tozeroy')])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        if not historico.empty:
            # Exibição segura da coluna confianca[span_3](start_span)[span_3](end_span)
            display_df = historico[['data', 'tipo', 'lucro', 'confianca']].copy()
            st.dataframe(display_df.head(15), use_container_width=True)
        else:
            st.info("Aguardando primeiro trade para gerar relatório.")

    with t3:
        try:
            translator = GoogleTranslator(source='en', target='pt')
            feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
            for n in feed.entries[:6]:
                st.markdown(f'<div class="news-card"><b>{translator.translate(n.title)}</b><br><small>{n.published}</small></div>', unsafe_allow_html=True)
        except:
            st.write("Erro ao carregar notícias. Verifique a conexão.")

except Exception as e:
    st.error(f"Erro Crítico: {e}")
    if st.button("Resetar Banco de Dados"):
        import os
        if os.path.exists('cotton_intel.db'): os.remove('cotton_intel.db')
        st.rerun()
