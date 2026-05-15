import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import sqlite3
import pytz

# 1. CONFIGURAÇÃO DE SEGURANÇA E PERFORMANCE
# Atualização a cada 60 segundos para evitar bloqueios (Rate Limit)
st_autorefresh(interval=60 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Estável", layout="centered")

# --- BANCO DE DADOS (Persistência de Saldo e Trades) ---
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

def get_saldo():
    conn = sqlite3.connect('cotton_intel.db')
    res = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    conn.close()
    return res

# --- ESTILO VISUAL ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-box { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; border: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px; }
    .card { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #deff9a; margin-bottom: 10px; }
    .stMetric { background-color: #1c2128; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE HORÁRIO DE MERCADO ---
def verificar_mercado():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    # Segunda a Sexta, das 10h às 17h (Horário aproximado NY/B3)
    if agora.weekday() >= 5: 
        return "🔴 MERCADO FECHADO (FIM DE SEMANA)", "#4a1010"
    if 10 <= agora.hour < 17:
        return "🟢 MERCADO ABERTO (AO VIVO)", "#104a10"
    else:
        return "🟡 AFTER-MARKET / FECHADO", "#4a4110"

# --- PROCESSAMENTO DE DADOS E IA ---
@st.cache_data(ttl=300) # Mantém dados por 5 min para evitar erro de "Too Many Requests"
def carregar_dados():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    # Busca dados históricos
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # Indicadores Fundamentais e Técnicos
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    df['USDA_Estoque'] = 76.4
    df['Spread'] = df['Algodao'] / df['Petroleo']
    df['Weather_Risk'] = 6.8
    
    # Preparação da IA
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread', 'Weather_Risk']
    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(df[features][:-1], df['Target'][:-1])
    
    return modelo, df, features

# --- EXECUÇÃO DO APP ---
try:
    modelo, dados, features = carregar_dados()
    preco_atual = dados['Algodao'].iloc[-1]
    msg_m, cor_m = verificar_mercado()
    
    # 1. STATUS NO TOPO
    st.markdown(f'<div class="status-box" style="background-color: {cor_m};">{msg_m}</div>', unsafe_allow_html=True)
    st.title("🌱 Cotton Intelligence")

    # 2. PAINEL DE FUNDAMENTOS
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f'<div class="card"><small>USDA (ESTOQUE)</small><br><b style="font-size:20px;">76.4M Fardos</b><br><span style="color:#deff9a;">📉 Tendência Baixa</span></div>', unsafe_allow_html=True)
    with col_info2:
        st.markdown(f'<div class="card" style="border-left-color: #ff4b4b;"><small>CLIMA (TEXAS)</small><br><b style="font-size:20px;">SECA D4</b><br><span style="color:#ff4b4b;">⚠️ Risco de Safra</span></div>', unsafe_allow_html=True)

    # 3. MÉTRICAS EM TEMPO REAL
    c1, c2, c3 = st.columns(3)
    c1.metric("COT. ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR (DXY)", f"{dados['Dolar'].iloc[-1]:.1f}")

    st.markdown("---")

    # 4. INTELIGÊNCIA ARTIFICIAL (DINÂMICA)
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    # Cores dinâmicas baseadas na confiança
    if prob > 0.65:
        cor_ia, txt_ia = "#deff9a", "SINAL DE COMPRA FORTE"
    elif prob > 0.45:
        cor_ia, txt_ia = "#fccf03", "MERCADO NEUTRO / AGUARDAR"
    else:
        cor_ia, txt_ia = "#ff4b4b", "RISCO DE BAIXA DETECTADO"

    st.markdown(f"**Confiança da IA:** <span style='color:{cor_ia}; font-size:22px;'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
    st.progress(prob)
    st.markdown(f"<p style='color:{cor_ia}; font-weight:bold;'>{txt_ia}</p>", unsafe_allow_html=True)

    # 5. PAINEL DE TRADING E SALDO
    saldo_permanente = get_saldo()
    st.sidebar.header("💰 Gestão de Carteira")
    st.sidebar.metric("Saldo Líquido", f"${saldo_permanente:,.2f}")
    
    st.subheader("Painel de Execução")
    qtd_contratos = st.number_input("Quantidade (Contratos):", min_value=1, value=100, step=50)
    
    col_buy, col_sell = st.columns(2)
    with col_buy:
        if st.button("🟢 EXECUTAR COMPRA", use_container_width=True):
            st.session_state.p_entrada = preco_atual
            st.session_state.p_qtd = qtd_contratos
            st.success(f"Compra de {qtd_contratos} efetuada!")
            
    with col_sell:
        if st.button("🔴 FECHAR POSIÇÃO", use_container_width=True):
            if 'p_entrada' in st.session_state:
                lucro_final = (preco_atual - st.session_state.p_entrada) * st.session_state.p_qtd
                # Atualiza Banco de Dados
                conn = sqlite3.connect('cotton_intel.db')
                conn.execute('UPDATE conta SET saldo = ? WHERE id = 1', (saldo_permanente + lucro_final,))
                conn.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro) VALUES (?,?,?,?,?)',
                             (datetime.now().strftime("%d/%m %H:%M"), "LONG", st.session_state.p_entrada, preco_atual, round(lucro_final, 2)))
                conn.commit()
                conn.close()
                del st.session_state.p_entrada
                st.rerun()
            else:
                st.warning("Nenhuma posição aberta.")

    # 6. ANÁLISE GRÁFICA E HISTÓRICO
    tab1, tab2 = st.tabs(["📊 Gráfico de Preços", "📁 Histórico de Trades"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dados['Algodao'].tail(60), name="Preço Atual", line=dict(color=cor_ia, width=3), fill='tozeroy'))
        fig.add_trace(go.Scatter(y=dados['MA20'].tail(60), name="Média 20 dias", line=dict(color='rgba(255,255,255,0.4)', dash='dot')))
        fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        conn = sqlite3.connect('cotton_intel.db')
        df_historico = pd.read_sql_query("SELECT data, tipo, entrada, saida, lucro FROM trades ORDER BY id DESC", conn)
        st.table(df_historico)
        conn.close()

except Exception as e:
    st.error(f"Erro de Conexão: {e}. O Yahoo Finance pode estar temporariamente indisponível. Aguarde 1 minuto.")

