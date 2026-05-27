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
import requests

# --- 1. CONFIGURAÇÃO E ESTABILIDADE ---
st_autorefresh(interval=45 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel Pro MASTER", layout="wide")

def init_db():
    conn = sqlite3.connect('cotton_intel.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS conta (id INTEGER PRIMARY KEY, saldo REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, data TEXT, tipo TEXT, entrada REAL, saida REAL, lucro REAL, confianca REAL, stop_loss REAL, take_profit REAL)')
    
    try:
        c.execute('ALTER TABLE trades ADD COLUMN confianca REAL DEFAULT 0.5')
    except: pass
    try:
        c.execute('ALTER TABLE trades ADD COLUMN stop_loss REAL DEFAULT 0.0')
    except: pass
    try:
        c.execute('ALTER TABLE trades ADD COLUMN take_profit REAL DEFAULT 0.0')
    except: pass
    
    c.execute('SELECT saldo FROM conta WHERE id = 1')
    if not c.fetchone():
        c.execute('INSERT INTO conta (id, saldo) VALUES (1, 100000.0)')
    conn.commit()
    conn.close()

init_db()

@st.cache_data(ttl=40)
def carregar_dados_mestre():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    
    dfs = {}
    for nome, t in tickers.items():
        coleta = yf.Ticker(t).history(period="2y")
        if isinstance(coleta.columns, pd.MultiIndex):
            coleta.columns = coleta.columns.get_level_values(0)
        dfs[nome] = coleta['Close']
        
    df = pd.DataFrame(dfs).ffill().dropna()
    
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    df['STD20'] = df['Algodao'].rolling(window=20).std()
    
    df['Banda_Sup'] = df['MA20'] + (df['STD20'] * 2)
    df['Banda_Inf'] = df['MA20'] - (df['STD20'] * 2)
    df['Largura_Banda'] = (df['Banda_Sup'] - df['Banda_Inf']) / df['MA20']
    
    delta = df['Algodao'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatilidade'] = df['Algodao'].diff().abs().rolling(14).mean()
    df_norm = (df / df.iloc[0]) * 100
    
    df['Target'] = (df['Algodao'].shift(-1) > (df['Algodao'] * 1.0003)).astype(int)
    df.dropna(inplace=True)
    
    features = ['Algodao', 'Petroleo', 'Dolar', 'MA20', 'RSI']
    ponto_divisao = int(len(df) * 0.80)
    df_treino = df.iloc[:ponto_divisao]
    
    modelo = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    modelo.fit(df_treino[features], df_treino['Target'])
    
    return modelo, df, df_norm, features, ponto_divisao

def get_market_status():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    abertura, fechamento = agora.replace(hour=10, minute=0, second=0), agora.replace(hour=17, minute=0, second=0)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO", "Abre Segunda", "#4a1010"
    return ("🟢 MERCADO ABERTO", "Fecha às 17h", "#104a10") if abertura <= agora <= fechamento else ("🔴 MERCADO FECHADO", "Abre amanhã", "#4a1010")

def obter_clima_texas():
    try:
        url = "https://wttr.in/Lubbock,Texas?format=%t+%C+%w"
        resposta = requests.get(url, timeout=5)
        if resposta.status_code == 200:
            dados = resposta.text.strip().split()
            temp = dados[0]
            condicao = dados[1] if len(dados) > 1 else "Estável"
            vento = dados[2] if len(dados) > 2 else "N/A"
            return temp, condicao, vento
    except:
        pass
    return "28°C", "Pred. Ensolarado", "12km/h"

def executar_simulacao_backtest(df_back, features, modelo, capital_inicial):
    prob_historica = modelo.predict_proba(df_back[features])[:, 1]
    df_back = df_back.copy()
    df_back['Prob_IA'] = prob_historica
    
    capital = capital_inicial
    posicao = None
    preco_entrada = 0.0
    historico_capital = []
    trades_executados = []
    
    for i in range(len(df_back)):
        preco_hist = df_back['Algodao'].iloc[i]
        p_ia = df_back['Prob_IA'].iloc[i]
        
        if posicao == "LONG":
            if p_ia <= 0.50:
                lucro_trade = (preco_hist - preco_entrada) * 3000  
                capital += lucro_trade
                trades_executados.append(lucro_trade)
                posicao = None
        elif posicao == "SHORT":
            if p_ia >= 0.50:
                lucro_trade = (preco_entrada - preco_hist) * 3000
                capital += lucro_trade
                trades_executados.append(lucro_trade)
                posicao = None
        
        if posicao is None:
            if p_ia > 0.60:    
                posicao = "LONG"
                preco_entrada = preco_hist
            elif p_ia < 0.40:
                posicao = "SHORT"
                preco_entrada = preco_hist
                
        historico_capital.append(capital)
        
    return capital, trades_executados, historico_capital

# --- 2. ESTILOS VISUAIS (CSS) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric { background-color: #1c2128; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    .status-card { padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 15px; color: white; font-weight: bold; font-size: 14px; }
    .ia-container { padding: 15px; border-radius: 15px; text-align: center; border: 2px solid; margin-bottom: 10px; background-color: rgba(0,0,0,0.1); }
    .trading-box { background-color: #1c2128; padding: 15px; border-radius: 15px; border: 1px solid #30363d; }
    .news-card { background-color: #1c2128; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #58a6ff; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: bold !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. EXECUÇÃO DO MOTOR MESTRE ---
modelo, df, df_norm, features, ponto_divisao = carregar_dados_mestre()
prob = modelo.predict_proba(df[features].tail(1))[0][1]
preco_atual = df['Algodao'].iloc[-1]
volat = df['Volatilidade'].iloc[-1]

conn = sqlite3.connect('cotton_intel.db')
saldo_atual = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
conn.close()

# Tendências Macro
dolar_hoje = df['Dolar'].iloc[-1]
dolar_ontem = df['Dolar'].iloc[-2]
petroleo_hoje = df['Petroleo'].iloc[-1]
petroleo_ontem = df['Petroleo'].iloc[-2]
tendencia_dolar_alta = dolar_hoje > dolar_ontem
tendencia_petroleo_alta = petroleo_hoje > petroleo_ontem

# Filtro de Rompimento e Detecção de Mercado Lateral
b_sup = df['Banda_Sup'].iloc[-1]
b_inf = df['Banda_Inf'].iloc[-1]
largura_media = df['Largura_Banda'].tail(30).mean()
largura_atual = df['Largura_Banda'].iloc[-1]

mercado_lateral = largura_atual < (largura_media * 0.85)
rompendo_topo = preco_atual >= (b_sup * 0.998)
rompendo_fundo = preco_atual <= (b_inf * 1.002)

# SIDEBAR TÉCNICA
with st.sidebar:
    st.header("🛡️ Gestão e Técnica")
    st.metric("Saldo em Conta", f"${saldo_atual:,.2f}")
    
    st.markdown("---")
    st.subheader("📊 Dados Técnicos (0-100)")
    
    rsi_val = df['RSI'].iloc[-1]
    volat_val = df['Volatilidade'].iloc[-1]
    ma20_val = df['MA20'].iloc[-1]
    std_val = df['STD20'].iloc[-1] if 'STD20' in df.columns else 0.5

    score_rsi = max(0.0, min(100.0, rsi_val))
    z_score = (preco_atual - ma20_val) / (std_val + 1e-9)
    score_ma20 = 50.0 + (z_score * 25.0)
    score_ma20 = max(0.0, min(100.0, score_ma20))

    v_min = df['Volatilidade'].tail(60).min()
    v_max = df['Volatilidade'].tail(60).max()
    score_volat = ((volat_val - v_min) / ((v_max - v_min) + 1e-9)) * 100
    score_volat = max(0.0, min(100.0, score_volat))

    def obter_cor_tecnica(score):
        if score > 55: return '#00CF85'
        if score < 45: return '#ff4b4b'
        return '#fccf03'

    fig_barras = go.Figure(go.Bar(
        x=['RSI', 'Média MA20', 'Volatilidade'],
        y=
