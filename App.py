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
    c.execute('CREATE TABLE IF NOT EXISTS conta (id INTEGER PRIMARY KEY, saldo REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, data TEXT, tipo TEXT, entrada REAL, saida REAL, lucro REAL, confianca REAL)')
    try:
        c.execute('ALTER TABLE trades ADD COLUMN confianca REAL DEFAULT 0.5')
    except:
        pass 
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
        p_alta = ['rise', 'high', 'shortage', 'bullish', 'increase', 'drought', 'demand']
        p_baixa = ['fall', 'low', 'surplus', 'bearish', 'decrease', 'oversupply', 'drop']
        for n in feed.entries[:10]:
            t = n.title.lower()
            if any(w in t for w in p_alta): score += 1
            if any(w in t for w in p_baixa): score -= 1
        return 1 if score > 0 else -1 if score < 0 else 0
    except: return 0

@st.cache_data(ttl=40)
def carregar_dados_mestre():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="2y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # Indicadores Técnicos[span_2](start_span)[span_2](end_span)
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

# --- 2. ESTILO CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; margin-bottom: 10px; }
    .ia-container { padding: 20px; border-radius: 15px; text-align: center; border: 2px solid; margin-bottom: 10px; background-color: rgba(0,0,0,0.2); }
    .side-monitor { background-color: #1c2128; padding: 15px; border-radius: 10px; border: 1px solid #30363d; height: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LÓGICA DE INTERFACE ---
try:
    modelo, df, df_norm, features = carregar_dados_mestre()
    prob = modelo.predict_proba(df[features].tail(1))[0][1]
    preco_atual = df['Algodao'].iloc[-1]
    
    conn = sqlite3.connect('cotton_intel.db')
    saldo_atual = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    conn.close()

    # Barra Lateral Esquerda (Gestão)
    with st.sidebar:
        st.header("🛡️ Gestão")
        st.metric("Saldo Líquido", f"${saldo_atual:,.2f}")
        st.markdown("---")
        if 'ent' in st.session_state:
            st.warning("Posição em Aberto")
            if st.button("FECHAR POSIÇÃO"):
                # Lógica de fechamento simplificada para o exemplo
                del st.session_state.ent
                st.rerun()

    # LAYOUT PRINCIPAL: 3 Colunas[span_3](start_span)[span_3](end_span)
    # col_main: IA e Gráfico | col_side: Monitor Dólar/Petróleo
    col_main, col_side = st.columns([3, 1])

    with col_main:
        # Área da IA
        cor_ia, txt_ia = ("#deff9a", "COMPRA FORTE") if prob > 0.75 else ("#ff4b4b", "VENDA FORTE") if prob < 0.25 else ("#fccf03", "AGUARDAR")
        st.markdown(f'<div class="ia-container" style="border-color: {cor_ia}; color: {cor_ia};"><small style="color:white">PROBABILIDADE IA</small><br><span style="font-size: 40px; font-weight: 900;">{prob*100:.1f}%</span> - <b>{txt_ia}</b></div>', unsafe_allow_html=True)
        
        # Gráfico de Área (O que você tinha antes)[span_4](start_span)[span_4](end_span)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df['Algodao'].tail(60), 
            fill='tozeroy', 
            line=dict(color=cor_ia, width=3),
            name="Preço Algodão"
        ))
        fig.update_layout(
            template="plotly_dark", 
            height=450, 
            margin=dict(l=0,r=0,t=0,b=0),
            yaxis=dict(gridcolor='#30363d'),
            xaxis=dict(gridcolor='#30363d')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        # Aba Lateral Direita (Monitor Macro)[span_5](start_span)[span_5](end_span)
        st.markdown('<div class="side-monitor">', unsafe_allow_html=True)
        st.subheader("🌐 Monitor Macro")
        
        # Métricas de Petróleo e Dólar separadas à direita[span_6](start_span)[span_6](end_span)
        petroleo_val = df['Petroleo'].iloc[-1]
        dolar_val = df['Dolar'].iloc[-1]
        
        st.metric("PETRÓLEO (WTI)", f"${petroleo_val:.2f}", 
                  delta=f"{((petroleo_val/df['Petroleo'].iloc[-2])-1)*100:.2f}%")
        
        st.metric("DÓLAR (DXY)", f"{dolar_val:.2f}", 
                  delta=f"{((dolar_val/df['Dolar'].iloc[-2])-1)*100:.2f}%", delta_color="inverse")
        
        st.markdown("---")
        st.write("**Sentimento:**")
        sent = analisar_sentimento_noticias()
        st.info("📈 Otimista" if sent == 1 else "📉 Pessimista" if sent == -1 else "⚖️ Neutro")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Abas Inferiores
    t1, t2 = st.tabs(["📜 Histórico", "📰 Notícias"])
    with t1:
        conn = sqlite3.connect('cotton_intel.db')
        hist = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC LIMIT 5", conn)
        conn.close()
        st.table(hist)
    with t2:
        st.caption("Notícias mundiais traduzidas em tempo real...")

except Exception as e:
    st.error(f"Erro ao carregar layout: {e}")
