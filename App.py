
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# 1. CONFIGURAÇÃO
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Data Management", layout="centered")

# --- SISTEMA DE GESTÃO DE DADOS (OPÇÕES 1 E 2) ---
if 'saldo' not in st.session_state: st.session_state.saldo = 100000.0  
if 'posicao' not in st.session_state: st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state: st.session_state.preco_entrada = 0.0
# Opção 2: Histórico para Gráfico de Performance
if 'historico_patrimonio' not in st.session_state: st.session_state.historico_patrimonio = [100000.0]
# Opção 1: Log Detalhado para Exportação CSV
if 'log_trades' not in st.session_state: st.session_state.log_trades = []

# --- STATUS DO MERCADO ---
def verificar_mercado():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO"
    return "🟢 MERCADO ABERTO" if 10 <= agora.hour < 17 else "🟡 AFTER-MARKET"

@st.cache_data(ttl=0)
def carregar_ia():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    # Parâmetros dos 4 Passos (USDA, Correlação, COT, Clima)
    df['USDA_Estoque'], df['Spread_Petroleo'], df['COT_Sentiment'], df['Weather_Risk'] = 76.4, df['Algodao']/df['Petroleo'], 1, 6.8
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread_Petroleo', 'COT_Sentiment', 'Weather_Risk']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_ia()
    preco_atual = dados['Algodao'].iloc[-1]
    
    st.title("🌱 Cotton Intel | Gestão de Dados")
    st.info(verificar_mercado())

    # --- MÉTRICAS E PAINEL LATERAL ---
    st.sidebar.header("💰 Saldo em Conta")
    st.sidebar.metric("Líquido", f"${st.session_state.saldo:.2f}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

    # --- OPERAÇÕES ---
    st.markdown("### Execução")
    qtd = st.number_input("Contratos:", min_value=1, value=100, step=50)
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        if st.button("🟢 COMPRAR"):
            st.session_state.saldo -= preco_atual * qtd
            st.session_state.posicao += qtd
            st.session_state.preco_entrada = preco_atual
            st.rerun()
            
    with col_b2:
        if st.button("🔴 VENDER (Fechar Pos)") :
            if st.session_state.posicao > 0:
                resultado_financeiro = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
                st.session_state.saldo += preco_atual * st.session_state.posicao
                
                # EXECUÇÃO DA OPÇÃO 1 (Gravar no Banco de Dados)
                novo_registo = {
                    "Data/Hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "Ativo": "Algodão (CT=F)",
                    "Preço Entrada": round(st.session_state.preco_entrada, 2),
                    "Preço Saída": round(preco_atual, 2),
                    "Lote": st.session_state.posicao,
                    "Resultado ($)": round(resultado_financeiro, 2)
                }
                st.session_state.log_trades.append(novo_registo)
                
                # EXECUÇÃO DA OPÇÃO 2 (Atualizar Curva de Património)
                st.session_state.historico_patrimonio.append(st.session_state.saldo)
                
                st.session_state.posicao = 0
                st.rerun()

    # --- INTERFACE DE GESTÃO (ABAS) ---
    tab_trade, tab_dados, tab_perf = st.tabs(["📊 Terminal", "📁 Base de Dados (CSV)", "📈 Performance"])

    with tab_trade:
        prob = modelo.predict_proba(dados[features].tail(1))[0][1]
        st.write(f"**Confiança da IA:** {prob*100:.1f}%")
        fig = go.Figure(go.Scatter(y=dados['Algodao'].tail(30), fill='tozeroy', line=dict(color='#deff9a')))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_dados:
        st.subheader("Opção 1: Log de Auditoria")
        if st.session_state.log_trades:
            df_log = pd.DataFrame(st.session_state.log_trades)
            st.table(df_log) # Exibe em formato de tabela limpa
            
            # Botão de Exportação
            csv_data = df_log.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar para Excel (CSV)",
                data=csv_data,
                file_name=f"log_algodao_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Nenhum dado de operação registado ainda.")

    with tab_perf:
        st.subheader("Opção 2: Curva de Equity")
        fig_equity = go.Figure(go.Scatter(y=st.session_state.historico_patrimonio, mode='lines+markers', line=dict(color='#deff9a', width=3)))
        fig_equity.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_equity, use_container_width=True)
        st.metric("Total de Operações", len(st.session_state.log_trades))

except Exception as e:
    st.error(f"Erro no sistema de dados: {e}")
