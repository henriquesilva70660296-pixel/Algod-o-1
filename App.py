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

# MOTOR 1: DADOS DIÁRIOS (MANTIDO INTACTO)
@st.cache_data(ttl=40)
def carregar_dados_mestre_diario():
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
    
    return modelo, df, df_norm, features

# MOTOR 2: NOVO MOTOR EXCLUSIVO PARA SINAIS DE 1 HORA
@st.cache_data(ttl=45)
def carregar_dados_mestre_1h():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {}
    for nome, t in tickers.items():
        coleta = yf.Ticker(t).history(period="730d", interval="1h")
        if isinstance(coleta.columns, pd.MultiIndex):
            coleta.columns = coleta.columns.get_level_values(0)
        dfs[nome] = coleta['Close']
        
    df = pd.DataFrame(dfs).ffill().dropna()
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    df['STD20'] = df['Algodao'].rolling(window=20).std()
    
    delta = df['Algodao'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Target'] = (df['Algodao'].shift(-1) > (df['Algodao'] * 1.0002)).astype(int)
    df.dropna(inplace=True)
    
    features = ['Algodao', 'Petroleo', 'Dolar', 'MA20', 'RSI']
    ponto_divisao = int(len(df) * 0.85)
    df_treino = df.iloc[:ponto_divisao]
    
    modelo_1h = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42)
    modelo_1h.fit(df_treino[features], df_treino['Target'])
    
    return modelo_1h, df, features

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
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric { background-color: #1c2128; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    .status-card { padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 15px; color: white; font-weight: bold; font-size: 14px; }
    .ia-container { padding: 12px; border-radius: 12px; text-align: center; border: 2px solid; margin-bottom: 10px; background-color: rgba(0,0,0,0.1); }
    .trading-box { background-color: #1c2128; padding: 15px; border-radius: 15px; border: 1px solid #30363d; }
    .news-card { background-color: #1c2128; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #58a6ff; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: bold !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. EXECUÇÃO DOS MOTORES ---
modelo_diario, df, df_norm, features_diario = carregar_dados_mestre_diario()
modelo_1h, df_1h, features_1h = carregar_dados_mestre_1h()

prob_diaria = modelo_diario.predict_proba(df[features_diario].tail(1))[0][1]
prob_1h = modelo_1h.predict_proba(df_1h[features_1h].tail(1))[0][1]

preco_atual = df['Algodao'].iloc[-1]
volat = df['Volatilidade'].iloc[-1]

conn = sqlite3.connect('cotton_intel.db')
saldo_atual = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
conn.close()

# Tendências Macro (Diárias)
dolar_hoje = df['Dolar'].iloc[-1]
dolar_ontem = df['Dolar'].iloc[-2]
petroleo_hoje = df['Petroleo'].iloc[-1]
petroleo_ontem = df['Petroleo'].iloc[-2]
tendencia_dolar_alta = dolar_hoje > dolar_ontem
tendencia_petroleo_alta = petroleo_hoje > petroleo_ontem

# Canais Diários (Mantidos idênticos)
b_sup = df['Banda_Sup'].iloc[-1]
b_inf = df['Banda_Inf'].iloc[-1]
largura_media = df['Largura_Banda'].tail(30).mean()
largura_atual = df['Largura_Banda'].iloc[-1]

mercado_lateral = largura_atual < (largura_media * 0.85)
rompendo_topo = preco_atual >= (b_sup * 0.998)
rompendo_fundo = preco_atual <= (b_inf * 1.002)

# SIDEBAR TÉCNICA (MANTIDA DIÁRIA)
with st.sidebar:
    st.header("🛡️ Gestão e Técnica")
    st.metric("Saldo em Conta", f"${saldo_atual:,.2f}")
    
    st.markdown("---")
    st.subheader("📊 Dados Técnicos Diários (0-100)")
    
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
        y=[score_rsi, score_ma20, score_volat],
        marker_color=[obter_cor_tecnica(score_rsi), obter_cor_tecnica(score_ma20), obter_cor_tecnica(score_volat)],
        text=[f"{score_rsi:.0f}", f"{score_ma20:.0f}", f"{score_volat:.0f}"],
        textposition='auto'
    ))
    fig_barras.update_layout(
        template="plotly_dark", height=180, margin=dict(l=5, r=5, t=5, b=5),
        yaxis=dict(range=[0, 100], gridcolor="#30363d"), showlegend=False
    )
    st.plotly_chart(fig_barras, use_container_width=True, config={'displayModeBar': False})
    
    st.subheader("📝 Valores Reais Brutos")
    st.write(f"RSI (14): **{rsi_val:.2f}**")
    st.write(f"Volatilidade: **{volat_val:.4f}**")
    st.write(f"Média (MA20): **{ma20_val:.4f}**")
    
    st.markdown("---")
    risco_p = st.slider("Risco Operação %", 0.5, 5.0, 2.0)
    lote = int((saldo_atual * (risco_p/100)) / ((volat * 2) * 100)) if volat > 0 else 5
    st.caption(f"Lote Sugerido: {lote} Ct")

# PAINEL PRINCIPAL
st_lab, t_lab, color = get_market_status()
st.markdown(f'<div class="status-card" style="background-color: {color};">{st_lab} | {t_lab}</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("COT. ALGODÃO", f"${preco_atual:.4f}")
m2.metric("PETRÓLEO", f"${petroleo_hoje:.2f}")
m3.metric("DÓLAR (DXY)", f"{dolar_hoje:.2f}")

st.markdown("---")

col_ia, col_trade = st.columns([1.5, 1])

with col_ia:
    st.subheader("🤖 Sinais de Inteligência Artificial")
    
    # --- LOGICA SINAL DIÁRIO (MANTIDO EXATAMENTE IGUAL) ---
    if mercado_lateral and (0.35 <= prob_diaria <= 0.65):
        cor_dia, txt_dia = "#fccf03", "MERCADO LATERAL"
    elif prob_diaria > 0.65:
        if tendencia_dolar_alta: cor_dia, txt_dia = "#fccf03", "COMPRA RISCO (DXY ▲)"
        elif rompendo_topo: cor_dia, txt_dia = "#00CF85", "⚡ BREAKOUT ALTA ⚡"
        else: cor_dia, txt_dia = "#00CF85", "COMPRA FORTE"
    elif prob_diaria < 0.35:
        if tendencia_petroleo_alta: cor_dia, txt_dia = "#fccf03", "VENDA RISCO (PETRÓLEO ▲)"
        elif rompendo_fundo: cor_dia, txt_dia = "#ff4b4b", "⚡ BREAKOUT BAIXA ⚡"
        else: cor_dia, txt_dia = "#ff4b4b", "VENDA FORTE"
    else:
        cor_dia, txt_dia = "#fccf03", "AGUARDAR"

    st.markdown(f"""
        <div class="ia-container" style="border-color: {cor_dia}; color: {cor_dia};">
            <small style="color: white; opacity: 0.6;">SINAL CANAL DIÁRIO (SWING)</small><br>
            <span style="font-size: 28px; font-weight: 800;">{prob_diaria*100:.1f}%</span> - <b>{txt_dia}</b>
        </div>
        """, unsafe_allow_html=True)

    # --- NOVA LÓGICA: SINAL DO GRÁFICO DE 1 HORA ---
    if 0.38 <= prob_1h <= 0.62:
        cor_h1, txt_h1 = "#fccf03", "LATERAL / INTRADAY"
    elif prob_1h > 0.62:
        cor_h1, txt_h1 = "#00CF85", "COMPRA RAPIDA (1H)"
    else:
        cor_h1, txt_h1 = "#ff4b4b", "VENDA RAPIDA (1H)"

    st.markdown(f"""
        <div class="ia-container" style="border-color: {cor_h1}; color: {cor_h1}; margin-top: 15px;">
            <small style="color: white; opacity: 0.6;">SINAL ESPECÍFICO GRÁFICO 1 HORA (INTRADAY)</small><br>
            <span style="font-size: 28px; font-weight: 800;">{prob_1h*100:.1f}%</span> - <b>{txt_h1}</b>
        </div>
        """, unsafe_allow_html=True)

with col_trade:
    st.markdown('<div class="trading-box">', unsafe_allow_html=True)
    if 'ent' not in st.session_state:
        qtd = st.number_input("Quantidade:", 1, 5000, lote, key="trade_q")
        tp_input = st.number_input("Take Profit (Alvo Ganho $):", 0.10, 10.00, 1.00, step=0.10, key="trade_tp")
        sl_input = st.number_input("Stop Loss (Limite Perda $):", 0.10, 5.00, 0.50, step=0.10, key="trade_sl")
        
        if st.button("🟢 EXECUTAR COMPRA", use_container_width=True):
            st.session_state.ent = preco_atual
            st.session_state.q = qtd
            st.session_state.tipo = "LONG"
            st.session_state.tp = preco_atual + tp_input
            st.session_state.sl = preco_atual - sl_input
            st.rerun()
        if st.button("🔴 EXECUTAR VENDA", use_container_width=True):
            st.session_state.ent = preco_atual
            st.session_state.q = qtd
            st.session_state.tipo = "SHORT"
            st.session_state.tp = preco_atual - tp_input
            st.session_state.sl = preco_atual + sl_input
            st.rerun()
    else:
        mult = 1 if st.session_state.tipo == "LONG" else -1
        lucro_v = (preco_atual - st.session_state.ent) * st.session_state.q * mult
        
        gatilho_fechamento = False
        if st.session_state.tipo == "LONG":
            if preco_atual >= st.session_state.tp or preco_atual <= st.session_state.sl: gatilho_fechamento = True
        elif st.session_state.tipo == "SHORT":
            if preco_atual <= st.session_state.tp or preco_atual >= st.session_state.sl: gatilho_fechamento = True
        
        if gatilho_fechamento:
            c = sqlite3.connect('cotton_intel.db')
            c.execute('UPDATE conta SET saldo = saldo + ?', (lucro_v,))
            c.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro, confianca, stop_loss, take_profit) VALUES (?,?,?,?,?,?,?,?)',
                     (datetime.now().strftime("%d/%m %H:%M"), st.session_state.tipo, st.session_state.ent, preco_atual, lucro_v, prob_diaria, st.session_state.sl, st.session_state.tp))
            c.commit(); c.close()
            del st.session_state.ent
            st.rerun()
        
        st.metric(f"Posição {st.session_state.tipo}", f"${lucro_v:,.2f}")
        if st.button("✖️ FECHAR POSIÇÃO MANUAL", use_container_width=True):
            c = sqlite3.connect('cotton_intel.db')
            c.execute('UPDATE conta SET saldo = saldo + ?', (lucro_v,))
            c.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro, confianca, stop_loss, take_profit) VALUES (?,?,?,?,?,?,?,?)',
                     (datetime.now().strftime("%d/%m %H:%M"), st.session_state.tipo, st.session_state.ent, preco_atual, lucro_v, prob_diaria, st.session_state.sl, st.session_state.tp))
            c.commit(); c.close()
            del st.session_state.ent
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ABAS OPERACIONAIS
tab_g, tab_f, tab_c, tab_n = st.tabs(["📊 Gráficos", "📦 Fundamentos", "🔗 Macro", "📰 Radar"])

with tab_g:
    st.subheader("⏱️ Gráfico do Algodão (Tempo Real / 1m)")
    try:
        dados_vapt = yf.download(tickers="CT=F", period="1d", interval="1m", progress=False)
        if isinstance(dados_vapt.columns, pd.MultiIndex):
            dados_vapt.columns = dados_vapt.columns.get_level_values(0)
        dados_vapt = dados_vapt.reset_index()
        
        if not dados_vapt.empty:
            eixo_x_g = dados_vapt['Datetime'] if 'Datetime' in dados_vapt.columns else dados_vapt['Date']
            fig_minuto = go.Figure(go.Scatter(
                x=eixo_x_g, y=dados_vapt['Close'], mode='lines', 
                line=dict(color='#00CF85', width=2), name='Preço'
            ))
            fig_minuto.update_layout(
                template="plotly_dark", height=200, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(type='category', tickangle=0, nticks=4), yaxis=dict(gridcolor="#30363d")
            )
            st.plotly_chart(fig_minuto, use_container_width=True, config={'displayModeBar': False})
    except:
        st.caption("Sincronizando feed minuto a minuto...")

    st.markdown("---")
    st.subheader("🗓️ Histórico Diário com Bandas de Bollinger (Seus Canais)")
    df_rec = df.tail(45)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['Algodao'], line=dict(color='#58a6ff', width=2), name='Preço Diário'))
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['MA20'], line=dict(color='#ff9f43', width=1.5, dash='dash'), name='Média MA20'))
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['Banda_Sup'], line=dict(color='rgba(255,255,255,0.2)', width=1), name='Banda Sup'))
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['Banda_Inf'], line=dict(color='rgba(255,255,255,0.2)', width=1), name='Banda Inf', fill='tonexty', fillcolor='rgba(255,255,255,0.03)'))
    
    fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tab_f:
    st.subheader("🌾 Clima em Tempo Real - Polo Produtor (Lubbock, Texas)")
    temp_tx, cond_tx, vento_tx = obter_clima_texas()
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="stMetric"><b>TEMPERATURA</b><br>{temp_tx}<br><small>Estresse Térmico</small></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stMetric"><b>CONDIÇÃO</b><br>{cond_tx}<br><small>Impacto na Lavoura</small></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stMetric"><b>VENTOS</b><br>{vento_tx}<br><small>Dispersão</small></div>', unsafe_allow_html=True)

with tab_c:
    st.subheader("🔗 Correlação Macro Histórica (2 Anos)")
    fig_c = go.Figure()
    for col in ["Algodao", "Petroleo", "Dolar"]: fig_c.add_trace(go.Scatter(y=df_norm[col], name=col))
    fig_c.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})

with tab_n:
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
        translator = GoogleTranslator(source='en', target='pt')
        for n in feed.entries[:4]:
            try: st.markdown(f'<div class="news-card"><small>{n.published}</small><br><b>{translator.translate(n.title)}</b></div>', unsafe_allow_html=True)
            except: st.write(n.title)
    except: st.write("Radar em atualização...")
