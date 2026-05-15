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

# --- 2. LÓGICA DE HORÁRIO DINÂMICO ---
def get_market_status():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    abertura = agora.replace(hour=10, minute=0, second=0)
    fechamento = agora.replace(hour=17, minute=0, second=0)
    
    if agora.weekday() >= 5:
        return "🔴 MERCADO FECHADO (FIM DE SEMANA)", "Abre Segunda às 10h", "#4a1010"
    
    if agora < abertura:
        falta = abertura - agora
        return "🟡 PRÉ-MERCADO", f"Abre em {str(falta).split('.')[0]}", "#4a4110"
    elif agora > fechamento:
        return "🔴 MERCADO FECHADO", "Abre Amanhã às 10h", "#4a1010"
    else:
        falta = fechamento - agora
        return "🟢 MERCADO ABERTO", f"Fecha em {str(falta).split('.')[0]}", "#104a10"

# --- 3. ESTILO CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .status-card { padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; color: white; }
    .ia-box { padding: 20px; border-radius: 15px; text-align: center; border: 2px solid; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. EXECUÇÃO ---
try:
    modelo, df, df_norm, features = carregar_dados_mestre()
    preco_atual = df['Algodao'].iloc[-1]
    volat = df['Volatilidade'].iloc[-1]
    saldo_atual = sqlite3.connect('cotton_intel.db').execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]

    # SIDEBAR: GESTÃO DE RISCO
    with st.sidebar:
        st.title("🛡️ Risk Control")
        st.metric("Saldo em Conta", f"${saldo_atual:,.2f}")
        st.markdown("---")
        risco_perc = st.slider("Risco por Operação (%)", 0.5, 5.0, 2.0)
        risco_financeiro = saldo_atual * (risco_perc / 100)
        lote_sugerido = int(risco_financeiro / ((volat * 2) * 100)) if volat > 0 else 100
        
        st.write(f"Arriscar: **${risco_financeiro:,.2f}**")
        st.write(f"Lote Sugerido: **{lote_sugerido} Ct**")
        
        st.markdown("---")
        if 'ent' in st.session_state:
            st.warning(f"Posição Aberta em: ${st.session_state.ent:.2f}")
            if st.button("🔴 FECHAR AGORA", use_container_width=True):
                lucro = (preco_atual - st.session_state.ent) * st.session_state.q
                conn = sqlite3.connect('cotton_intel.db')
                conn.execute('UPDATE conta SET saldo = saldo + ?', (lucro,))
                conn.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro) VALUES (?,?,?,?,?)',
                             (datetime.now().strftime("%d/%m %H:%M"), "LONG", st.session_state.ent, preco_atual, lucro))
                conn.commit()
                conn.close()
                del st.session_state.ent
                st.rerun()

    # TOPO: STATUS E MÉTRICAS
    status_label, tempo_label, cor_fundo = get_market_status()
    st.markdown(f"""
        <div class="status-card" style="background-color: {cor_fundo};">
            <span style="font-size: 20px; font-weight: bold;">{status_label}</span><br>
            <span style="opacity: 0.8;">{tempo_label}</span>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("COT. ALGODÃO", f"${preco_atual:.4f}")
    c2.metric("PETRÓLEO", f"${df['Petroleo'].iloc[-1]:.2f}")
    c3.metric("DÓLAR (DXY)", f"{df['Dolar'].iloc[-1]:.2f}")

    # ÁREA CENTRAL: IA DINÂMICA
    prob = modelo.predict_proba(df[features].tail(1))[0][1]
    if prob > 0.65:
        cor_ia, txt_ia, sub_ia = "#deff9a", "FORTE TENDÊNCIA DE ALTA", "Probabilidade elevada de lucro em Compra"
    elif prob < 0.35:
        cor_ia, txt_ia, sub_ia = "#ff4b4b", "FORTE TENDÊNCIA DE BAIXA", "Risco elevado. Considere aguardar fora."
    else:
        cor_ia, txt_ia, sub_ia = "#fccf03", "MERCADO INDECISO", "IA detectou neutralidade. Cuidado com o volume."

    st.markdown(f"""
        <div class="ia-box" style="border-color: {cor_ia}; color: {cor_ia}; background-color: rgba(0,0,0,0.2);">
            <small style="color: white; opacity: 0.6;">CONFIANÇA DA INTELIGÊNCIA ARTIFICIAL</small><br>
            <span style="font-size: 45px; font-weight: 900;">{prob*100:.1f}%</span><br>
            <b style="font-size: 18px;">{txt_ia}</b><br>
            <span style="color: white; font-size: 14px;">{sub_ia}</span>
        </div>
        """, unsafe_allow_html=True)

    # ABAS ORGANIZADAS
    t_graf, t_fund, t_corr, t_news = st.tabs(["📊 Gráfico e Ordens", "📦 Fundamentos (USDA/Clima)", "🔗 Correlação", "📰 Radar"])

    with t_graf:
        fig = go.Figure(go.Scatter(y=df['Algodao'].tail(60), line=dict(color=cor_ia, width=3), fill='tozeroy'))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        qtd_op = st.number_input("Quantidade para operar:", 1, 10000, lote_sugerido)
        if st.button("🟢 EXECUTAR COMPRA NO PREÇO ATUAL", use_container_width=True):
            st.session_state.ent = preco_atual
            st.session_state.q = qtd_op
            st.balloons()

    with t_fund:
        f1, f2 = st.columns(2)
        f1.markdown('<div class="stMetric" style="border-left: 5px solid #0088ff;"><b>ESTOQUE USDA</b><br>76.4M Fardos<br><small>Próximo relatório: WASDE Junho</small></div>', unsafe_allow_html=True)
        f2.markdown('<div class="stMetric" style="border-left: 5px solid #ff4b4b;"><b>CLIMA (TEXAS)</b><br>Seca Severa (D4)<br><small>Risco de abandono de safra elevado</small></div>', unsafe_allow_html=True)

    with t_corr:
        fig_c = go.Figure()
        for c in df_norm.columns:
            fig_c.add_trace(go.Scatter(x=df_norm.index, y=df_norm[c], name=c))
        fig_c.update_layout(template="plotly_dark", height=400, title="Performance Normalizada")
        st.plotly_chart(fig_c, use_container_width=True)

    with t_news:
        url = "https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for n in feed.entries[:8]:
            st.markdown(f'**{n.published}** - [{n.title}]({n.link})')

except Exception as e:
    st.info("Sincronizando dados com a bolsa... Aguarde um instante.")

