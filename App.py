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

# --- 2. ESTILO CSS ---
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

# --- 3. EXECUÇÃO ---
try:
    modelo, df, df_norm, features, ponto_divisao = carregar_dados_mestre()
    prob = modelo.predict_proba(df[features].tail(1))[0][1]
    preco_atual = df['Algodao'].iloc[-1]
    volat = df['Volatilidade'].iloc[-1]
    
    conn = sqlite3.connect('cotton_intel.db')
    saldo_atual = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    conn.close()

    # SIDEBAR
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
    m2.metric("PETRÓLEO", f"${df['Petroleo'].iloc[-1]:.2f}")
    m3.metric("DÓLAR (DXY)", f"{df['Dolar'].iloc[-1]:.2f}")

    st.markdown("---")

    col_ia, col_trade = st.columns([1.5, 1])

    with col_ia:
        cor_ia, txt_ia = ("#deff9a", "COMPRA FORTE") if prob > 0.65 else ("#ff4b4b", "VENDA FORTE") if prob < 0.35 else ("#fccf03", "AGUARDAR")
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
            qtd = st.number_input("Quantidade:", 1, 5000, lote, key="trade_q")
            tp_input = st.number_input("Take Profit (Alvo Ganho $):", 0.10, 10.00, 1.00, step=0.10, key="trade_tp")
            sl_input = st.number_input("Stop Loss (Limite Perda $):", 0.10, 5.00, 0.50, step=0.10, key="trade_sl")
            
            # SEUS BOTÕES ORIGINAIS DO SEU LAYOUT PREFERIDO
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
                         (datetime.now().strftime("%d/%m %H:%M"), st.session_state.tipo, st.session_state.ent, preco_atual, lucro_v, prob, st.session_state.sl, st.session_state.tp))
                c.commit(); c.close()
                del st.session_state.ent
                st.rerun()
            
            st.metric(f"Posição {st.session_state.tipo}", f"${lucro_v:,.2f}")
            if st.button("✖️ FECHAR POSIÇÃO MANUAL", use_container_width=True):
                c = sqlite3.connect('cotton_intel.db')
                c.execute('UPDATE conta SET saldo = saldo + ?', (lucro_v,))
                c.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro, confianca, stop_loss, take_profit) VALUES (?,?,?,?,?,?,?,?)',
                         (datetime.now().strftime("%d/%m %H:%M"), st.session_state.tipo, st.session_state.ent, preco_atual, lucro_v, prob, st.session_state.sl, st.session_state.tp))
                c.commit(); c.close()
                del st.session_state.ent
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ABAS OPERACIONAIS
    tab_g, tab_f, tab_c, tab_n, tab_b = st.tabs(["📊 Gráfico", "📦 Fundamentos", "🔗 Macro", "📰 Radar", "📈 Backtest IA"])

    with tab_g:
        st.subheader("⏱️ Gráfico do Algodão (Tempo Real / 1m)")
        try:
            dados_vapt = yf.download(tickers="CT=F", period="1d", interval="1m")
            if isinstance(dados_vapt.columns, pd.MultiIndex):
                dados_vapt.columns = dados_vapt.columns.get_level_values(0)
            dados_vapt = dados_vapt.reset_index()
            
            if dados_vapt.empty or len(dados_vapt) < 2:
                dados_vapt = yf.download(tickers="CT=F", period="5d", interval="30m")
                if isinstance(dados_vapt.columns, pd.MultiIndex):
                    dados_vapt.columns = dados_vapt.columns.get_level_values(0)
                dados_vapt = dados_vapt.reset_index()
            
            if not dados_vapt.empty:
                fig_minuto = go.Figure(go.Scatter(
                    x=dados_vapt['Datetime'] if 'Datetime' in dados_vapt.columns else dados_vapt['Date'], 
                    y=dados_vapt['Close'], mode='lines', 
                    line=dict(color='#00CF85', width=2), name='Preço'
                ))
                fig_minuto.update_layout(
                    template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(type='category', tickangle=0, nticks=4)
                )
                st.plotly_chart(fig_minuto, use_container_width=True)
            else:
                st.caption("Aguardando novas oscilações...")
        except:
            st.caption("Sincronizando feed de cotações...")

        st.markdown("---")
        st.subheader("🗓️ Histórico de Médio Prazo (Média Móvel 20)")
        dados_preco = df['Algodao'].tail(45)
        dados_ma20 = df['MA20'].tail(45)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dados_preco.index, y=dados_preco, line=dict(color='#58a6ff', width=2), name='Preço'))
        fig.add_trace(go.Scatter(x=dados_ma20.index, y=dados_ma20, line=dict(color='#ff9f43', width=1.5, dash='dash'), name='MA20'))
        fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_f:
        f1, f2 = st.columns(2)
        f1.markdown('<div class="stMetric"><b>ESTOQUE USDA</b><br>76.4M Fardos<br><small>Fonte: WASDE</small></div>', unsafe_allow_html=True)
        f2.markdown('<div class="stMetric"><b>VOLATILIDADE</b><br>Alta (HVT)<br><small>Foco: Texas/EUA</small></div>', unsafe_allow_html=True)

    with tab_c:
        st.subheader("🔗 Correlação Macro em Tempo Real (Hoje / 1m)")
        try:
            tickers_fast = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
            dfs_fast = {}
            for nome, t in tickers_fast.items():
                coleta_f = yf.download(tickers=t, period="1d", interval="1m")
                if isinstance(coleta_f.columns, pd.MultiIndex):
                    coleta_f.columns = coleta_f.columns.get_level_values(0)
                dfs_fast[nome] = coleta_f['Close']
                
            df_fast = pd.DataFrame(dfs_fast).ffill().dropna().reset_index()

            if not df_fast.empty:
                # Normalização dinâmica baseada no primeiro candle do dia
                eixo_x = df_fast['Datetime'] if 'Datetime' in df_fast.columns else df_fast['Date']
                df_fast_calc = df_fast[["Algodao", "Petroleo", "Dolar"]]
                df_fast_norm = (df_fast_calc / df_fast_calc.iloc[0]) * 100
                
                fig_c_fast = go.Figure()
                colors_fast = {"Algodao": "#00CF85", "Petroleo": "#ff9f43", "Dolar": "#54a0ff"}
                for col in df_fast_norm.columns:
                    fig_c_fast.add_trace(go.Scatter(
                        x=eixo_x, y=df_fast_norm[col], 
                        name=col, line=dict(color=colors_fast.get(col), width=2)
                    ))
                fig_c_fast.update_layout(
                    template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(type='category', tickangle=0, nticks=4), legend=dict(orientation="h", y=1.1, x=0)
                )
                st.plotly_chart(fig_c_fast, use_container_width=True)
            else:
                st.caption("Aguardando novas oscilações do mercado macro...")
        except Exception as e:
            st.caption("Sincronizando fluxo macro de alta frequência...")

        st.markdown("---")
        st.subheader("🔗 Correlação Macro Histórica (2 Anos)")
        fig_c = go.Figure()
        for col in df_norm.columns: 
            fig_c.add_trace(go.Scatter(y=df_norm[col], name=col))
        fig_c.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_c, use_container_width=True)

    with tab_n:
        feed = feedparser.parse("https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en")
        translator = GoogleTranslator(source='en', target='pt')
        for n in feed.entries[:4]:
            try: st.markdown(f'<div class="news-card"><small>{n.published}</small><br><b>{translator.translate(n.title)}</b></div>', unsafe_allow_html=True)
            except: st.write(n.title)

    with tab_b:
        st.subheader("🕵️ Simulação com Dados Inéditos")
        df_back = df.iloc[ponto_divisao:].copy()
        prob_historica = modelo.predict_proba(df_back[features])[:, 1]
        df_back['Prob_IA'] = prob_historica
        
        capital_inicial = 100000.0
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
            
        lucro_total_back = capital - capital_inicial
        total_trades = len(trades_executados)
        vitorias = sum(1 for t in trades_executados if t > 0)
        taxa_acerto = (vitorias / total_trades * 100) if total_trades > 0 else 0.0
        
        b1, b2 = st.columns(2)
        b1.metric("Retorno Real", f"${lucro_total_back:+,.2f}", delta=f"{(lucro_total_back/capital_inicial)*100:+.2f}%")
        b2.metric("Acertos", f"{taxa_acerto:.1f}%", f"{vitorias}W / {total_trades - vitorias}L")
        
        fig_back = go.Figure(go.Scatter(x=df_back.index, y=historico_capital, line=dict(color='#00CF85', width=2.5)))
        fig_back.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_back, use_container_width=True)

except Exception as e:
    st.error(f"Sincronizando motores: {e}")
