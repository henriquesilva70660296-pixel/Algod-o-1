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

@st.cache_data(ttl=40)
def carregar_dados_mestre():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="2y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # Inteligência v3.1
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    delta = df['Algodao'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Volatilidade'] = df['Algodao'].diff().abs().rolling(14).mean()
    df_norm = (df / df.iloc[0]) * 100
    df['Target'] = (df['Algodao'].shift(-1) > (df['Algodao'] * 1.0005)).astype(int)
    
    features = ['Algodao', 'Petroleo', 'Dolar', 'MA20', 'RSI']
    modelo = RandomForestClassifier(n_estimators=300, random_state=42).fit(df[features][:-1], df['Target'][:-1])
    
    return modelo, df, df_norm, features

def get_market_status():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    abertura, fechamento = agora.replace(hour=10, minute=0, second=0), agora.replace(hour=17, minute=0, second=0)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO", "Abre Segunda", "#4a1010"
    return ("🟢 MERCADO ABERTO", "Fecha às 17h", "#104a10") if abertura <= agora <= fechamento else ("🔴 MERCADO FECHADO", "Abre amanhã", "#4a1010")

# --- 2. ESTILO CSS (Visual da Versão Anterior) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric { background-color: #1c2128; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    .status-card { padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 15px; color: white; font-weight: bold; }
    .ia-container { padding: 20px; border-radius: 15px; text-align: center; border: 2px solid; margin-bottom: 10px; background-color: rgba(0,0,0,0.1); }
    .trading-box { background-color: #1c2128; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    .news-card { background-color: #1c2128; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. EXECUÇÃO ---
try:
    modelo, df, df_norm, features = carregar_dados_mestre()
    prob = modelo.predict_proba(df[features].tail(1))[0][1]
    preco_atual = df['Algodao'].iloc[-1]
    volat = df['Volatilidade'].iloc[-1]
    
    conn = sqlite3.connect('cotton_intel.db')
    saldo_atual = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    conn.close()

    # SIDEBAR COM DADOS TÉCNICOS
    with st.sidebar:
        st.header("🛡️ Gestão e Técnica")
        st.metric("Saldo em Conta", f"${saldo_atual:,.2f}")
        
        st.markdown("---")
        st.subheader("📊 Dados Técnicos")
        st.write(f"RSI (14): **{df['RSI'].iloc[-1]:.2f}**")
        st.write(f"Volatilidade: **{df['Volatilidade'].iloc[-1]:.4f}**")
        st.write(f"Média (MA20): **{df['MA20'].iloc[-1]:.4f}**")
        
        st.markdown("---")
        risco_p = st.slider("Risco Operação %", 0.5, 5.0, 2.0)
        lote = int((saldo_atual * (risco_p/100)) / ((volat * 2) * 100)) if volat > 0 else 10
        st.caption(f"Lote Sugerido: {lote} Ct")

    # Status e Métricas de Topo
    st_lab, t_lab, color = get_market_status()
    st.markdown(f'<div class="status-card" style="background-color: {color};">{st_lab} | {t_lab}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("COT. ALGODÃO", f"${preco_atual:.4f}")
    m2.metric("PETRÓLEO", f"${df['Petroleo'].iloc[-1]:.2f}")
    m3.metric("DÓLAR (DXY)", f"{df['Dolar'].iloc[-1]:.2f}")

    st.markdown("---")

    # Área Central: IA e Boleta de Trade
    col_ia, col_trade = st.columns([1.5, 1])

    with col_ia:
        cor_ia, txt_ia = ("#deff9a", "COMPRA FORTE") if prob > 0.70 else ("#ff4b4b", "VENDA FORTE") if prob < 0.30 else ("#fccf03", "AGUARDAR")
        st.markdown(f"""
            <div class="ia-container" style="border-color: {cor_ia}; color: {cor_ia};">
                <small style="color: white; opacity: 0.6;">CONFIANÇA DA IA MASTER</small><br>
                <span style="font-size: 50px; font-weight: 900;">{prob*100:.1f}%</span><br>
                <b style="font-size: 20px;">{txt_ia}</b>
            </div>
            """, unsafe_allow_html=True)

    with col_trade:
        st.markdown('<div class="trading-box">', unsafe_allow_html=True)
        if 'ent' not in st.session_state:
            qtd = st.number_input("Quantidade:", 1, 5000, lote)
            if st.button("🟢 EXECUTAR COMPRA", use_container_width=True):
                st.session_state.ent, st.session_state.q, st.session_state.tipo = preco_atual, qtd, "LONG"
                st.rerun()
            if st.button("🔴 EXECUTAR VENDA", use_container_width=True):
                st.session_state.ent, st.session_state.q, st.session_state.tipo = preco_atual, qtd, "SHORT"
                st.rerun()
        else:
            mult = 1 if st.session_state.tipo == "LONG" else -1
            lucro_v = (preco_atual - st.session_state.ent) * st.session_state.q * mult
            st.metric(f"Posição {st.session_state.tipo}", f"${lucro_v:,.2f}", delta=f"{((preco_atual/st.session_state.ent)-1)*100*mult:.2f}%")
            if st.button("✖️ FECHAR POSIÇÃO", use_container_width=True):
                c = sqlite3.connect('cotton_intel.db')
                c.execute('UPDATE conta SET saldo = saldo + ?', (lucro_v,))
                c.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro, confianca) VALUES (?,?,?,?,?,?)',
                         (datetime.now().strftime("%d/%m %H:%M"), st.session_state.tipo, st.session_state.ent, preco_atual, lucro_v, prob))
                c.commit(); c.close()
                del st.session_state.ent
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Abas (Igual à versão solicitada)
    tab_g, tab_f, tab_c, tab_n = st.tabs(["📊 Gráfico", "📦 Fundamentos", "🔗 Macro", "📰 Radar"])

    with tab_g:
        # --- [MOVIDO PARA O TOPO] NOVO GRÁFICO EM TEMPO REAL MINUTO A MINUTO ---
        st.subheader("⏱️ Gráfico do Algodão em Tempo Real (1 Minuto)")
        try:
            dados_vapt = yf.download(tickers="CT=F", period="1d", interval="1m")
            if not dados_vapt.empty:
                fig_minuto = go.Figure(go.Scatter(
                    x=dados_vapt.index, 
                    y=dados_vapt['Close'], 
                    mode='lines+markers', 
                    line=dict(color='#00CF85', width=2),
                    name='Preço Rápido'
                ))
                fig_minuto.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_minuto, use_container_width=True)
            else:
                st.caption("Aguardando novas oscilações do mercado minuto a minuto...")
        except:
            st.caption("Conectando ao fluxo de dados rápidos...")

        # --- SEU GRÁFICO ORIGINAL (FICOU EM SEGUNDO LUGAR) ---
        st.markdown("---")
        st.subheader("🗓️ Histórico de Médio Prazo (60 Períodos)")
        fig = go.Figure(go.Scatter(y=df['Algodao'].tail(60), line=dict(color=cor_ia, width=3), fill='tozeroy'))
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_f:
        f1, f2 = st.columns(2)
        f1.markdown('<div class="stMetric"><b>ESTOQUE USDA</b><br>76.4M Fardos<br><small>Fonte: WASDE</small></div>', unsafe_allow_html=True)
        f2.markdown('<div class="stMetric"><b>VOLATILIDADE</b><br>Alta (HVT)<br><small>Foco: Texas/EUA</small></div>', unsafe_allow_html=True)

    with tab_c:
        # --- [MOVIDO PARA O TOPO] NOVO GRÁFICO DE CORRELAÇÃO NORMALIZADA EM TEMPO REAL (1 MINUTO) ---
        st.subheader("🔗 Correlação Normalizada em Tempo Real (Hoje, 1 Minuto)")
        try:
            tickers_fast = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
            dfs_fast = {nome: yf.download(tickers=t, period="1d", interval="1m")['Close'] for nome, t in tickers_fast.items()}
            df_fast = pd.DataFrame(dfs_fast).ffill().dropna()

            if not df_fast.empty:
                df_fast_norm = (df_fast / df_fast.iloc[0]) * 100
                
                fig_c_fast = go.Figure()
                colors_fast = {"Algodao": "#00CF85", "Petroleo": "#ff9f43", "Dolar": "#54a0ff"}
                
                for col in df_fast_norm.columns:
                    fig_c_fast.add_trace(go.Scatter(
                        x=df_fast_norm.index,
                        y=df_fast_norm[col],
                        name=f"{col} (1m)",
                        line=dict(color=colors_fast.get(col, None), width=2)
                    ))
                fig_c_fast.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_c_fast, use_container_width=True)
            else:
                st.caption("Aguardando abertura dos mercados para cruzar as correlações diárias...")
        except Exception as e:
            st.caption("Sincronizando fluxo macro de alta frequência...")

        # --- SEU GRÁFICO DE CORRELAÇÃO HISTÓRICA ORIGINAL (FICOU EM SEGUNDO LUGAR) ---
        st.markdown("---")
        fig_c = go.Figure()
        for col in df_norm.columns: fig_c.add_trace(go.Scatter(y=df_norm[col], name=col))
        fig_c.update_layout(template="plotly_dark", height=350, title="Correlação Normalizada Histórica (2 Anos)")
        st.plotly_chart(fig_c, use_container_width=True)

    with tab_n:
        feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
        translator = GoogleTranslator(source='en', target='pt')
        for n in feed.entries[:5]:
            try:
                st.markdown(f'<div class="news-card"><small>{n.published}</small><br><b>{translator.translate(n.title)}</b></div>', unsafe_allow_html=True)
            except:
                st.write(n.title)

except Exception as e:
    st.error(f"Sincronizando: {e}")
