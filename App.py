import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
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
    
    # Tratamento seguro de extração dos dados históricos de fechamento
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

# --- 2. ESTILOS VISUAIS (CSS) ---
st.markdown("""
    <style>
