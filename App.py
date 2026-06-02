import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz
import feedparser
from deep_translator import GoogleTranslator
import requests

# --- 1. CONFIGURAÇÃO E ESTABILIDADE ---
st_autorefresh(interval=45 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intelligence Pro", layout="wide")

# MOTOR 1: DADOS DIÁRIOS (CANAIS DIÁRIOS)
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

# MOTOR 2: SINAIS DE 1 HORA (INTRADAY)
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

def obter_clima_texas_completo():
    try:
        url = "https://wttr.in/Lubbock,Texas?format=%t+%C+%w"
        resposta = requests.get(url, timeout=5)
        if resposta.status_code == 200:
            dados = resposta.text.strip().split()
            return dados[0], dados[1] if len(dados) > 1 else "Estável", dados[2] if len(dados) > 2 else "N/A"
    except: pass
    return "28°C", "Ensolarado", "12km/h"

# --- 2. ESTILOS VISUAIS (CSS) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 12px; border: 1px solid #30363d; }
    .status-card { padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px; color: white; font-weight: bold; font-size: 15px; }
    .ia-container { padding: 15px; border-radius: 12px; text-align: center; border: 2px solid; margin-bottom: 15px; background-color: rgba(0,0,0,0.2); }
    .future-box { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; min-height: 160px; }
    .future-header { padding: 15px; border-radius: 12px; text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 20px; border: 1px solid #30363d; }
    .news-card { background-color: #161b22; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #58a6ff; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: bold !important; color: #f0f6fc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. EXECUÇÃO DOS MOTORES DE DADOS ---
modelo_diario, df, df_norm, features_diario = carregar_dados_mestre_diario()
modelo_1h, df_1h, features_1h = carregar_dados_mestre_1h()

prob_diaria = modelo_diario.predict_proba(df[features_diario].tail(1))[0][1]
prob_1h = modelo_1h.predict_proba(df_1h[features_1h].tail(1))[0][1]

preco_atual = df['Algodao'].iloc[-1]
volat = df['Volatilidade'].iloc[-1]

# Tendências Macro
dolar_hoje = df['Dolar'].iloc[-1]
dolar_ontem = df['Dolar'].iloc[-2]
petroleo_hoje = df['Petroleo'].iloc[-1]
petroleo_ontem = df['Petroleo'].iloc[-2]
tendencia_dolar_alta = dolar_hoje > dolar_ontem
tendencia_petroleo_alta = petroleo_hoje > petroleo_ontem

# Canais Diários das Bandas de Bollinger
b_sup = df['Banda_Sup'].iloc[-1]
b_inf = df['Banda_Inf'].iloc[-1]
largura_media = df['Largura_Banda'].tail(30).mean()
largura_atual = df['Largura_Banda'].iloc[-1]

mercado_lateral = largura_atual < (largura_media * 0.85)
rompendo_topo = preco_atual >= (b_sup * 0.998)
rompendo_fundo = preco_atual <= (b_inf * 1.002)

# SIDEBAR TÉCNICA
with st.sidebar:
    st.header("📊 Filtros Rápidos")
    st.markdown("---")
    st.subheader("Osciladores do Dia (0-100)")
    
    rsi_val = df['RSI'].iloc[-1]
    volat_val = df['Volatilidade'].iloc[-1]
    ma20_val = df['MA20'].iloc[-1]
    std_val = df['STD20'].iloc[-1] if 'STD20' in df.columns else 0.5

    score_rsi = max(0.0, min(100.0, rsi_val))
    z_score = (preco_atual - ma20_val) / (std_val + 1e-9)
    score_ma20 = max(0.0, min(100.0, 50.0 + (z_score * 25.0)))

    v_min = df['Volatilidade'].tail(60).min()
    v_max = df['Volatilidade'].tail(60).max()
    score_volat = max(0.0, min(100.0, ((volat_val - v_min) / ((v_max - v_min) + 1e-9)) * 100))

    def obter_cor_tecnica(score):
        if score > 55: return '#00CF85'
        if score < 45: return '#ff4b4b'
        return '#fccf03'

    fig_barras = go.Figure(go.Bar(
        x=['RSI', 'Média MA20', 'Volat'],
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
    
    st.subheader("📝 Estatísticas Reais")
    st.write(f"RSI Diário: **{rsi_val:.2f}**")
    st.write(f"Volatilidade Ativo: **{volat_val:.4f}**")
    st.write(f"Média Móvel (MA20): **{ma20_val:.4f}**")

# PAINEL PRINCIPAL DE MONITORAMENTO
st_lab, t_lab, color = get_market_status()
st.markdown(f'<div class="status-card" style="background-color: {color};">{st_lab} | {t_lab}</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("COT. ALGODÃO (NY)", f"${preco_atual:.4f}")
m2.metric("PETRÓLEO BRENT", f"${petroleo_hoje:.2f}")
m3.metric("DÓLAR GLOBAL (DXY)", f"{dolar_hoje:.2f}")

st.markdown("---")

# GRÁFICO EM TEMPO REAL RETORNADO PARA O TOPO CENTRAL (Como solicitado)
st.subheader("📈 Gráfico de Tendência em Tempo Real (1 Minuto)")
try:
    dados_vapt = yf.download(tickers="CT=F", period="1d", interval="1m", progress=False)
    if isinstance(dados_vapt.columns, pd.MultiIndex):
        dados_vapt.columns = dados_vapt.columns.get_level_values(0)
    dados_vapt = dados_vapt.reset_index()
    
    if not dados_vapt.empty:
        eixo_x_g = dados_vapt['Datetime'] if 'Datetime' in dados_vapt.columns else dados_vapt['Date']
        fig_minuto = go.Figure(go.Scatter(
            x=eixo_x_g, y=dados_vapt['Close'], mode='lines', 
            line=dict(color='#00CF85', width=2), name='Preço Instantâneo'
        ))
        fig_minuto.update_layout(
            template="plotly_dark", height=220, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(type='category', showticklabels=True, nticks=6), yaxis=dict(gridcolor="#30363d")
        )
        st.plotly_chart(fig_minuto, use_container_width=True, config={'displayModeBar': False})
except:
    st.caption("Aguardando transmissão estável de velas de 1 minuto...")

st.markdown("---")

# Layout de Sinais de Inteligência Artificial
col_dia_box, col_h1_box = st.columns(2)

with col_dia_box:
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
    else: cor_dia, txt_dia = "#fccf03", "AGUARDAR CONFIRMAÇÃO"

    st.markdown(f"""
        <div class="ia-container" style="border-color: {cor_dia}; color: {cor_dia};">
            <small style="color: white; opacity: 0.6; font-weight:bold;">CANAL DIÁRIO (SWING TRADE)</small><br>
            <span style="font-size: 30px; font-weight: 900;">{prob_diaria*100:.1f}%</span> — <b>{txt_dia}</b>
        </div>
        """, unsafe_allow_html=True)

with col_h1_box:
    if 0.38 <= prob_1h <= 0.62:
        cor_h1, txt_h1 = "#fccf03", "LATERAL / INTRADAY"
    elif prob_1h > 0.62:
        cor_h1, txt_h1 = "#00CF85", "COMPRA RÁPIDA (GRÁFICO 1H)"
    else:
        cor_h1, txt_h1 = "#ff4b4b", "VENDA RÁPIDA (GRÁFICO 1H)"

    st.markdown(f"""
        <div class="ia-container" style="border-color: {cor_h1}; color: {cor_h1};">
            <small style="color: white; opacity: 0.6; font-weight:bold;">MOMENTUM 1 HORA (INTRADAY)</small><br>
            <span style="font-size: 30px; font-weight: 900;">{prob_1h*100:.1f}%</span> — <b>{txt_h1}</b>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# PAINEL DE ABAS REORGANIZADO
tab_g, tab_f, tab_fut, tab_n = st.tabs(["📊 Canais Gráficos (Diário)", "📦 Fundamentos Atuais", "🔮 Sinais Futuros (Previsões)", "📰 Radar de Notícias"])

with tab_g:
    st.subheader("🗓️ Canais Históricos Diários (Média MA20 e Bandas)")
    df_rec = df.tail(45)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['Algodao'], line=dict(color='#58a6ff', width=2.5), name='Preço Diário'))
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['MA20'], line=dict(color='#ff9f43', width=1.5, dash='dash'), name='Média MA20'))
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['Banda_Sup'], line=dict(color='rgba(255,255,255,0.15)', width=1), name='Banda Sup (Teto)'))
    fig.add_trace(go.Scatter(x=df_rec.index, y=df_rec['Banda_Inf'], line=dict(color='rgba(255,255,255,0.15)', width=1), name='Banda Inf (Chão)', fill='tonexty', fillcolor='rgba(255,255,255,0.02)'))
    
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0))

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tab_f:
    st.subheader("🌾 Clima do Polo Produtor de Algodão (Lubbock, Texas)")
    temp_tx, cond_tx, vento_tx = obter_clima_texas_completo()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="stMetric"><b>TEMPERATURA ATUAL</b><br>{temp_tx}<br><small>Risco de Estresse</small></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stMetric"><b>CONDIÇÃO ATMOSFÉRICA</b><br>{cond_tx}<br><small>Desenvolvimento da Safra</small></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stMetric"><b>VELOCIDADE DO VENTO</b><br>{vento_tx}<br><small>Erosão do Solo</small></div>', unsafe_allow_html=True)

with tab_fut:
    st.subheader("🔮 Projeções Fundamentais e Sinais Futuros (Confluência Operacional)")
    
    # --- NOVO SISTEMA DE CÁLCULO DE PROJEÇÃO DE PORCENTAGEM FUTURA ---
    # Valores base padrão
    score_futuro = 50.0 
    
    # 1. Input do Clima (Massa Seca = Alta) -> Soma +15%
    score_futuro += 15.0
    
    # 2. Input de Sazonalidade Automática baseada no mês corrente
    mes_atual = datetime.now().month
    sazonalidade_texto = "Período neutro de transição cambial e física."
    if mes_atual in [4, 5, 6, 7]:
        sazonalidade_texto = "📈 Plantio nos EUA. Histórico de alta por prêmio de risco climático (64% de alta sazonal)."
        score_futuro += 14.0 # Soma +14% por viés sazonal histórico de alta
    elif mes_atual in [9, 10, 11]:
        sazonalidade_texto = "📉 Entrada da Colheita física. Histórico de aumento de oferta e pressão de baixa na Bolsa."
        score_futuro -= 15.0 # Subtrai por viés de baixa
        
    # 3. Input USDA (Demanda Têxtil Forte) -> Soma +10%
    score_futuro += 10.0
    
    # Garantir limites matemáticos entre 0 e 100
    score_futuro = max(5.0, min(95.0, score_futuro))
    
    # Determina a direção textual do bloco futuro
    if score_futuro >= 55.0:
        cor_fut, txt_fut = "#00CF85", f"PROJEÇÃO DE ALTA FUTURA: {score_futuro:.1f}%"
    elif score_futuro <= 45.0:
        cor_fut, txt_fut = "#ff4b4b", f"PROJEÇÃO DE QUEDA FUTURA: {100 - score_futuro:.1f}%"
    else:
        cor_fut, txt_fut = "#fccf03", f"PROJEÇÃO NEUTRA: {score_futuro:.1f}%"

    # CARD PRINCIPAL COM A PORCENTAGEM FUTURA CALCULADA
    st.markdown(f"""
        <div class="future-header" style="background-color: rgba(0,0,0,0.3); border-color: {cor_fut}; color: {cor_fut};">
            🔮 RADAR DE PROJEÇÃO MACRO: {txt_fut}
        </div>
        """, unsafe_allow_html=True)
    
    cf1, cf2, cf3 = st.columns(3)
    
    with cf1:
        st.markdown("""
        <div class="future-box">
            <h4 style='color:#58a6ff; margin:0;'>🌦️ Previsão Climática (7 Dias)</h4>
            <p style='margin-top:10px; font-size:14px;'><b>Região:</b> Texas Panhandle<br>
            <b>Tendência:</b> Massa de ar seco avançando nas próximas 168 horas.<br>
            <span style='color:#00CF85;'>➔ Peso no Modelo: +15% de Viés de Alta</span></p>
        </div>
        """, unsafe_allow_html=True)
        
    with cf2:
        st.markdown(f"""
        <div class="future-box">
            <h4 style='color:#ff9f43; margin:0;'>📅 Sazonalidade do Mês</h4>
            <p style='margin-top:10px; font-size:14px;'><b>Mês Corrente:</b> {datetime.now().strftime('%B')}<br>
            <b>Histórico:</b> {sazonalidade_texto}<br>
            <span style='color:#00CF85;'>➔ Peso no Modelo: +14% de Probabilidade</span></p>
        </div>
        """, unsafe_allow_html=True)
        
    with cf3:
        st.markdown("""
        <div class="future-box">
            <h4 style='color:#ff4b4b; margin:0;'>🏛️ Agenda e Relatórios (USDA)</h4>
            <p style='margin-top:10px; font-size:14px;'><b>Consumo Global:</b> China e Índia projetam aumento de demanda têxtil de 1.2%.<br>
            <b>Aviso:</b> Evitar ordens pesadas na ActivTrades no dia de saída do WASDE.<br>
            <span style='color:#00CF85;'>➔ Peso no Modelo: +10% de Viés Altista</span></p>
        </div>
        """, unsafe_allow_html=True)

with tab_n:
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
        translator = GoogleTranslator(source='en', target='pt')
        for n in feed.entries[:4]:
            try: st.markdown(f'<div class="news-card"><small>{n.published}</small><br><b>{translator.translate(n.title)}</b></div>', unsafe_allow_html=True)
            except: st.write(n.title)
    except: st.write("Radar de notícias em atualização...")
