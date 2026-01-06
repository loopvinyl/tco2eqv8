import requests
from bs4 import BeautifulSoup
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

np.random.seed(50)

st.set_page_config(page_title="Simulador de Emissões CO₂eq", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def obter_cotacao_carbono_investing():
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.investing.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        selectores = [
            '[data-test="instrument-price-last"]',
            '.text-2xl',
            '.last-price-value',
            '.instrument-price-last',
            '.pid-1062510-last',
            '.float_lang_base_1',
            '.top.bold.inlineblock',
            '#last_last'
        ]
        
        preco = None
        fonte = "Investing.com"
        
        for seletor in selectores:
            try:
                elemento = soup.select_one(seletor)
                if elemento:
                    texto_preco = elemento.text.strip().replace(',', '')
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        import re
        padroes_preco = [
            r'"last":"([\d,]+)"',
            r'data-last="([\d,]+)"',
            r'last_price["\']?:\s*["\']?([\d,]+)',
            r'value["\']?:\s*["\']?([\d,]+)'
        ]
        
        html_texto = str(soup)
        for padrao in padroes_preco:
            matches = re.findall(padrao, html_texto)
            for match in matches:
                try:
                    preco_texto = match.replace(',', '')
                    preco = float(preco_texto)
                    if 50 < preco < 200:
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except ValueError:
                    continue
                    
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    return 85.57, "€", "Carbon Emissions (EU ETS Reference)", False, "EU ETS Reference Price"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    return 6.36, "R$", False, "Reference Rate for EU ETS"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {st.session_state.taxa_cambio:.2f}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {preco_carbono_reais:.2f}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        preco_voluntario_usd = 7.48
        preco_regulado_eur = 85.57
        taxa_cambio_usd = 5.0
        taxa_cambio_eur = st.session_state.taxa_cambio
        
        preco_voluntario_brl = preco_voluntario_usd * taxa_cambio_usd
        preco_regulado_brl = preco_regulado_eur * taxa_cambio_eur
        
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {st.session_state.taxa_cambio:.2f}
        - **Carbono em Reais:** R$ {preco_carbono_reais:.2f}/tCO₂eq
        
        **🌍 Comparação de Mercados:**
        - **Mercado Voluntário (SOVCM):** USD {preco_voluntario_usd:.2f} ≈ R$ {preco_voluntario_brl:.2f}/tCO₂eq
        - **Mercado Regulado (EU ETS):** €{preco_regulado_eur:.2f} ≈ R$ {preco_regulado_brl:.2f}/tCO₂eq
        - **Diferença:** {preco_regulado_brl/preco_voluntario_brl:.1f}x maior no regulado
        """)

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False

inicializar_session_state()

def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format_5_dec(x, pos):
    return f"{x:,.5f}".replace(",", "X").replace(".", ",").replace("X", ".")

st.title("Simulador de Emissões de tCO₂eq com Análise Técnico-Econômica")
st.markdown("Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para dois contextos de gestão de resíduos.")

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", 
                               min_value=10, max_value=1000, value=100, step=10)
    
    st.subheader("📊 Parâmetros Operacionais")
    
    umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0
    st.write(f"**Umidade selecionada:** {formatar_br(umidade_valor)}%")
    
    massa_exposta_kg = st.slider("Massa exposta na frente de trabalho (kg)", 50, 200, 100, 10)
    h_exposta = st.slider("Horas expostas por dia", 4, 24, 8, 1)
    
    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)
    
    with st.expander("🏭 Parâmetros TEA (Análise Técnico-Econômica)"):
        st.markdown("#### 💼 Parâmetros de Custo")
        
        fator_capex = st.slider(
            "Fator de ajuste CAPEX", 
            0.5, 2.0, 1.0, 0.1,
            key="fator_capex"
        )
        
        fator_opex = st.slider(
            "Fator de ajuste OPEX", 
            0.5, 2.0, 1.0, 0.1,
            key="fator_opex"
        )
        
        st.markdown("#### 📈 Parâmetros de Mercado")
        
        mercado_carbono = st.selectbox(
            "Mercado de Carbono para Análise",
            ["Híbrido (Média)", "Voluntário (USD 7.48)", "Regulado (EU ETS €85.57)", "Customizado"],
            key="mercado_carbono"
        )
        
        if mercado_carbono == "Customizado":
            preco_carbono_custom = st.number_input(
                "Preço Customizado (R$/tCO₂eq)",
                min_value=0.0,
                value=290.82,
                step=10.0,
                key="preco_carbono_custom"
            )
        
        preco_humus = st.number_input(
            "Preço do Húmus (R$/kg)",
            min_value=0.5,
            value=2.5,
            step=0.1,
            key="preco_humus"
        )
        
        taxa_desconto = st.slider(
            "Taxa de desconto para VPL (%)",
            0.0, 20.0, 8.0, 0.5,
            key="taxa_desconto"
        ) / 100
        
        st.markdown("#### 📊 Custos de Referência")
        custo_aterro = st.number_input(
            "Custo de disposição em aterro (R$/kg)",
            min_value=0.05,
            value=0.15,
            step=0.01
        )
    
    if st.button("🚀 Executar Simulação Completa", type="primary"):
        st.session_state.run_simulation = True

T = 25
DOC = 0.15
DOCf_val = 0.0147 * T + 0.28
MCF = 1
F = 0.5
OX = 0.1
Ri = 0.0
k_ano = 0.06
TOC_YANG = 0.436
TN_YANG = 14.2 / 1000
CH4_C_FRAC_YANG = 0.13 / 100
N2O_N_FRAC_YANG = 0.92 / 100
DIAS_COMPOSTAGEM = 50

PERFIL_CH4_VERMI = np.array([
    0.02, 0.02, 0.02, 0.03, 0.03,
    0.04, 0.04, 0.05, 0.05, 0.06,
    0.07, 0.08, 0.09, 0.10, 0.09,
    0.08, 0.07, 0.06, 0.05, 0.04,
    0.03, 0.02, 0.02, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01,
    0.005, 0.005, 0.005, 0.005, 0.005,
    0.005, 0.005, 0.005, 0.005, 0.005,
    0.002, 0.002, 0.002, 0.002, 0.002,
    0.001, 0.001, 0.001, 0.001, 0.001
])
PERFIL_CH4_VERMI /= PERFIL_CH4_VERMI.sum()

PERFIL_N2O_VERMI = np.array([
    0.15, 0.10, 0.20, 0.05, 0.03,
    0.03, 0.03, 0.04, 0.05, 0.06,
    0.08, 0.09, 0.10, 0.08, 0.07,
    0.06, 0.05, 0.04, 0.03, 0.02,
    0.01, 0.01, 0.005, 0.005, 0.005,
    0.005, 0.005, 0.005, 0.005, 0.005,
    0.002, 0.002, 0.002, 0.002, 0.002,
    0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001
])
PERFIL_N2O_VERMI /= PERFIL_N2O_VERMI.sum()

CH4_pre_descarte_ugC_por_kg_h_media = 2.78
fator_conversao_C_para_CH4 = 16/12
CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000

N2O_pre_descarte_mgN_por_kg = 20.26
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000

PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

dias = anos_simulacao * 365
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)
datas = pd.date_range(start=data_inicio, periods=dias, freq='D')
PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
CH4_C_FRAC_THERMO = 0.006
N2O_N_FRAC_THERMO = 0.0196

PERFIL_CH4_THERMO = np.array([
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.12, 0.15, 0.18, 0.20, 0.18,
    0.15, 0.12, 0.10, 0.08, 0.06,
    0.05, 0.04, 0.03, 0.02, 0.02,
    0.01, 0.01, 0.01, 0.01, 0.01,
    0.005, 0.005, 0.005, 0.005, 0.005,
    0.002, 0.002, 0.002, 0.002, 0.002,
    0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001
])
PERFIL_CH4_THERMO /= PERFIL_CH4_THERMO.sum()

PERFIL_N2O_THERMO = np.array([
    0.10, 0.08, 0.15, 0.05, 0.03,
    0.04, 0.05, 0.07, 0.10, 0.12,
    0.15, 0.18, 0.20, 0.18, 0.15,
    0.12, 0.10, 0.08, 0.06, 0.05,
    0.04, 0.03, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01,
    0.005, 0.005, 0.005, 0.005, 0.005,
    0.002, 0.002, 0.002, 0.002, 0.002,
    0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001,
])
PERFIL_N2O_THERMO /= PERFIL_N2O_THERMO.sum()

def ajustar_emissoes_pre_descarte(O2_concentracao):
    ch4_ajustado = CH4_pre_descarte_g_por_kg_dia

    if O2_concentracao == 21:
        fator_n2o = 1.0
    elif O2_concentracao == 10:
        fator_n2o = 11.11 / 20.26
    elif O2_concentracao == 1:
        fator_n2o = 7.86 / 20.26
    else:
        fator_n2o = 1.0

    n2o_ajustado = N2O_pre_descarte_g_por_kg_dia * fator_n2o
    return ch4_ajustado, n2o_ajustado

def calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao=dias):
    ch4_ajustado, n2o_ajustado = ajustar_emissoes_pre_descarte(O2_concentracao)

    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * ch4_ajustado / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (
                    residuos_kg_dia * n2o_ajustado * fracao / 1000
                )

    return emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg

def calcular_emissoes_aterro(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val = params

    fator_umid = (1 - umidade_val) / (1 - 0.55)
    f_aberto = np.clip((massa_exposta_kg / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
    docf_calc = 0.0147 * temp_val + 0.28

    potencial_CH4_por_kg = doc_val * docf_calc * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg

    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario

    E_aberto = 1.91
    E_fechado = 2.15
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia

    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]

    O2_concentracao = 21
    emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg = calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao)

    total_ch4_aterro_kg = emissoes_CH4 + emissoes_CH4_pre_descarte_kg
    total_n2o_aterro_kg = emissoes_N2O + emissoes_N2O_pre_descarte_kg

    return total_ch4_aterro_kg, total_n2o_aterro_kg

def calcular_emissoes_vermi(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val = params
    fracao_ms = 1 - umidade_val
    
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_YANG * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_YANG * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_VERMI[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_VERMI[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_compostagem(params, dias_simulacao=dias, dias_compostagem=50):
    umidade, T, DOC = params
    fracao_ms = 1 - umidade
    
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_THERMO[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_THERMO[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def executar_simulacao_completa(parametros):
    umidade, T, DOC = parametros
    
    ch4_aterro, n2o_aterro = calcular_emissoes_aterro([umidade, T, DOC])
    ch4_vermi, n2o_vermi = calcular_emissoes_vermi([umidade, T, DOC])

    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_vermi_tco2eq = (ch4_vermi * GWP_CH4_20 + n2o_vermi * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_vermi_tco2eq.sum()
    return reducao_tco2eq

def executar_simulacao_unfccc(parametros):
    umidade, T, DOC = parametros

    ch4_aterro, n2o_aterro = calcular_emissoes_aterro([umidade, T, DOC])
    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000

    ch4_compost, n2o_compost = calcular_emissoes_compostagem([umidade, T, DOC], dias_simulacao=dias, dias_compostagem=50)
    total_compost_tco2eq = (ch4_compost * GWP_CH4_20 + n2o_compost * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_compost_tco2eq.sum()
    return reducao_tco2eq

def calcular_custos_capex_opex(residuos_kg_dia, anos_operacao):
    CAPEX_BASE_R_por_kg_dia = 1500
    OPEX_ANUAL_R_por_kg_dia = 250
    
    capex_total = residuos_kg_dia * CAPEX_BASE_R_por_kg_dia
    opex_anual = residuos_kg_dia * OPEX_ANUAL_R_por_kg_dia
    
    custo_minhocas = residuos_kg_dia * 80
    custo_reatores = residuos_kg_dia * 1200
    custo_instalacao = residuos_kg_dia * 220
    
    capex_detalhado = {
        'Minhocas e substrato': custo_minhocas,
        'Reatores e estruturas': custo_reatores,
        'Instalação e montagem': custo_instalacao,
        'Projeto e engenharia': capex_total * 0.1,
        'Imprevistos (15%)': capex_total * 0.15
    }
    
    opex_detalhado = {
        'Mão de obra operacional': opex_anual * 0.4,
        'Energia e água': opex_anual * 0.15,
        'Manutenção': opex_anual * 0.15,
        'Administrativo': opex_anual * 0.2,
        'Impostos e taxas': opex_anual * 0.1
    }
    
    return {
        'capex_total': capex_total,
        'opex_anual': opex_anual,
        'capex_detalhado': capex_detalhado,
        'opex_detalhado': opex_detalhado
    }

def calcular_receitas(residuos_kg_dia, reducao_anual_tco2eq, preco_carbono_r, mercado='hibrido', preco_humus=2.5, custo_aterro=0.15):
    producao_humus_kg_ano = residuos_kg_dia * 0.3 * 365
    
    receita_humus = producao_humus_kg_ano * preco_humus
    receita_carbono = reducao_anual_tco2eq * preco_carbono_r
    economia_aterro = residuos_kg_dia * 365 * custo_aterro
    
    return {
        'receita_total_anual': receita_humus + receita_carbono + economia_aterro,
        'receita_humus': receita_humus,
        'receita_carbono': receita_carbono,
        'economia_aterro': economia_aterro,
        'producao_humus': producao_humus_kg_ano,
        'preco_credito_usado': preco_carbono_r,
        'mercado_selecionado': mercado
    }

def calcular_indicadores_financeiros(capex, opex_anual, receita_anual, anos, taxa_desconto=0.08):
    fluxo_caixa = [-capex]
    for ano in range(1, anos + 1):
        fluxo_anual = receita_anual - opex_anual
        fluxo_caixa.append(fluxo_anual)
    
    vpl = 0
    for t, fc in enumerate(fluxo_caixa):
        vpl += fc / ((1 + taxa_desconto) ** t)
    
    try:
        tir = np.irr(fluxo_caixa)
    except:
        tir = None
    
    acumulado = 0
    payback_anos = None
    for t, fc in enumerate(fluxo_caixa):
        if t == 0:
            continue
        acumulado += fc
        if acumulado >= capex and payback_anos is None:
            payback_anos = t
    
    acumulado_desc = 0
    payback_desc_anos = None
    for t, fc in enumerate(fluxo_caixa):
        if t == 0:
            continue
        fc_desc = fc / ((1 + taxa_desconto) ** t)
        acumulado_desc += fc_desc
        if acumulado_desc >= capex and payback_desc_anos is None:
            payback_desc_anos = t
    
    if receita_anual > 0:
        custo_tonelada_evitada = capex / (anos * (receita_anual / 1000))
    else:
        custo_tonelada_evitada = 0
    
    return {
        'vpl': vpl,
        'tir': tir,
        'payback_anos': payback_anos,
        'payback_desc_anos': payback_desc_anos,
        'fluxo_caixa': fluxo_caixa,
        'custo_tonelada_evitada': custo_tonelada_evitada,
        'taxa_desconto': taxa_desconto
    }

def analise_sensibilidade_tea(residuos_kg_dia, reducao_anual_tco2eq, anos_simulacao, preco_humus=2.5, custo_aterro=0.15):
    custos = calcular_custos_capex_opex(residuos_kg_dia, anos_simulacao)
    
    cenarios = {
        'Otimista': {
            'capex_fator': 0.85,
            'opex_fator': 0.90,
            'receita_fator': 1.20,
            'preco_carbono': 544.23,
            'preco_humus_fator': 1.2,
            'custo_aterro_fator': 1.2
        },
        'Base': {
            'capex_fator': 1.0,
            'opex_fator': 1.0,
            'receita_fator': 1.0,
            'preco_carbono': 290.82,
            'preco_humus_fator': 1.0,
            'custo_aterro_fator': 1.0
        },
        'Pessimista': {
            'capex_fator': 1.15,
            'opex_fator': 1.10,
            'receita_fator': 0.85,
            'preco_carbono': 37.40,
            'preco_humus_fator': 0.8,
            'custo_aterro_fator': 0.8
        }
    }
    
    resultados = {}
    for cenario, params in cenarios.items():
        capex_ajustado = custos['capex_total'] * params['capex_fator']
        opex_ajustado = custos['opex_anual'] * params['opex_fator']
        
        preco_humus_ajustado = preco_humus * params['preco_humus_fator']
        custo_aterro_ajustado = custo_aterro * params['custo_aterro_fator']
        
        receitas_ajustadas = calcular_receitas(
            residuos_kg_dia, 
            reducao_anual_tco2eq,
            params['preco_carbono'],
            mercado='regulado' if params['preco_carbono'] > 500 else 'voluntario',
            preco_humus=preco_humus_ajustado,
            custo_aterro=custo_aterro_ajustado
        )
        
        receita_ajustada = receitas_ajustadas['receita_total_anual'] * params['receita_fator']
        
        indicadores = calcular_indicadores_financeiros(
            capex_ajustado, 
            opex_ajustado, 
            receita_ajustada,
            anos_simulacao
        )
        
        resultados[cenario] = {
            'capex': capex_ajustado,
            'opex_anual': opex_ajustado,
            'receita_anual': receita_ajustada,
            'indicadores': indicadores,
            'receitas_detalhadas': receitas_ajustadas
        }
    
    return resultados

def criar_dashboard_tea(analise_tea, resultados_sensibilidade):
    st.subheader("🏭 Análise Técnico-Econômica (TEA)")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumo Executivo",
        "💰 Fluxo de Caixa",
        "📈 Indicadores Financeiros",
        "🎯 Análise de Sensibilidade",
        "⚖️ Trade-off Econômico-Ambiental"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💼 Investimento (CAPEX)")
            st.metric(
                "Investimento Total",
                f"R$ {formatar_br(analise_tea['capex_total'])}",
                help="Custo total de implantação do sistema"
            )
            
            st.markdown("**Detalhamento do CAPEX:**")
            for item, valor in analise_tea['capex_detalhado'].items():
                st.caption(f"{item}: R$ {formatar_br(valor)}")
        
        with col2:
            st.markdown("#### 💰 Custos Anuais (OPEX)")
            st.metric(
                "Custo Operacional Anual",
                f"R$ {formatar_br(analise_tea['opex_anual'])}/ano",
                help="Custos de operação e manutenção anuais"
            )
            
            st.markdown("**Detalhamento do OPEX:**")
            for item, valor in analise_tea['opex_detalhado'].items():
                st.caption(f"{item}: R$ {formatar_br(valor)}/ano")
    
    with tab2:
        st.markdown("#### 📈 Projeção de Fluxo de Caixa")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        anos = list(range(0, len(analise_tea['indicadores']['fluxo_caixa'])))
        fluxo_acumulado = np.cumsum(analise_tea['indicadores']['fluxo_caixa'])
        
        ax.bar(anos, analise_tea['indicadores']['fluxo_caixa'], 
               alpha=0.6, label='Fluxo Anual', color='skyblue')
        ax.plot(anos, fluxo_acumulado, 'r-', linewidth=3, 
                label='Fluxo Acumulado', marker='o')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        if analise_tea['indicadores']['payback_anos']:
            pb_ano = analise_tea['indicadores']['payback_anos']
            ax.axvline(x=pb_ano, color='green', linestyle='--', 
                      label=f'Payback: {pb_ano} anos')
        
        ax.set_xlabel('Ano')
        ax.set_ylabel('Fluxo de Caixa (R$)')
        ax.set_title('Projeção de Fluxo de Caixa do Projeto')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig)
        
        df_fluxo = pd.DataFrame({
            'Ano': anos,
            'Fluxo Anual (R$)': analise_tea['indicadores']['fluxo_caixa'],
            'Fluxo Acumulado (R$)': fluxo_acumulado
        })
        st.dataframe(df_fluxo.style.format({
            'Fluxo Anual (R$)': 'R$ {:.2f}',
            'Fluxo Acumulado (R$)': 'R$ {:.2f}'
        }))
    
    with tab3:
        st.markdown("#### 📊 Indicadores de Viabilidade Financeira")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vpl = analise_tea['indicadores']['vpl']
            st.metric(
                "VPL (Valor Presente Líquido)",
                f"R$ {formatar_br(vpl)}",
                delta=None,
                help="Valor presente dos fluxos de caixa futuros"
            )
        
        with col2:
            tir = analise_tea['indicadores']['tir']
            if tir:
                st.metric(
                    "TIR (Taxa Interna de Retorno)",
                    f"{tir*100:.1f}%",
                    help="Taxa de retorno que iguala o VPL a zero"
                )
            else:
                st.metric("TIR", "N/A")
        
        with col3:
            payback = analise_tea['indicadores']['payback_anos']
            if payback:
                st.metric(
                    "Payback Simples",
                    f"{payback} anos",
                    help="Tempo para recuperar o investimento"
                )
            else:
                st.metric("Payback", "> período")
        
        with col4:
            payback_desc = analise_tea['indicadores']['payback_desc_anos']
            if payback_desc:
                st.metric(
                    "Payback Descontado",
                    f"{payback_desc} anos",
                    help="Payback considerando valor do dinheiro no tempo"
                )
            else:
                st.metric("Payback Desc.", "> período")
        
        st.markdown("#### 📉 Análise de Ponto de Equilíbrio")
        
        receita_anual = analise_tea['receitas']['receita_total_anual']
        custo_fixo = analise_tea['opex_anual']
        custo_variavel = receita_anual * 0.3
        
        if receita_anual > custo_variavel:
            ponto_equilibrio = custo_fixo / (receita_anual - custo_variavel) * 100
        else:
            ponto_equilibrio = 100
        
        st.metric(
            "Ponto de Equilíbrio",
            f"{ponto_equilibrio:.1f}%",
            help="Percentual da capacidade necessária para cobrir custos"
        )
    
    with tab4:
        st.markdown("#### 🎯 Análise de Sensibilidade Financeira")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        base = resultados_sensibilidade['Base']['indicadores']['vpl']
        otimista = resultados_sensibilidade['Otimista']['indicadores']['vpl']
        pessimista = resultados_sensibilidade['Pessimista']['indicadores']['vpl']
        
        impacto_otimista = ((otimista - base) / base) * 100 if base != 0 else 0
        impacto_pessimista = ((pessimista - base) / base) * 100 if base != 0 else 0
        
        parametros = ['Cenário Otimista', 'Cenário Pessimista']
        impactos = [impacto_otimista, impacto_pessimista]
        
        y_pos = np.arange(len(parametros))
        colors = ['green' if x > 0 else 'red' for x in impactos]
        
        ax.barh(y_pos, impactos, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(parametros)
        ax.set_xlabel('Impacto no VPL (%)')
        ax.set_title('Análise de Sensibilidade - Impacto no VPL')
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(impactos):
            ax.text(v + (1 if v > 0 else -10), i, f'{v:.1f}%', 
                   color='black', va='center', fontweight='bold')
        
        st.pyplot(fig)
        
        st.markdown("#### 📋 Cenários Financeiros Comparativos")
        
        dados_cenarios = []
        for cenario, dados in resultados_sensibilidade.items():
            dados_cenarios.append({
                'Cenário': cenario,
                'CAPEX (R$)': formatar_br(dados['capex']),
                'VPL (R$)': formatar_br(dados['indicadores']['vpl']),
                'TIR (%)': f"{dados['indicadores']['tir']*100:.1f}" if dados['indicadores']['tir'] else 'N/A',
                'Payback (anos)': dados['indicadores']['payback_anos'] or '>20',
                'ROI (%)': f"{(dados['indicadores']['vpl']/dados['capex'])*100:.1f}" if dados['capex'] > 0 else 'N/A'
            })
        
        df_cenarios = pd.DataFrame(dados_cenarios)
        st.dataframe(df_cenarios, use_container_width=True)
    
    with tab5:
        st.markdown("#### ⚖️ Análise Custo-Benefício Ambiental")
        
        custo_tonelada = analise_tea['indicadores']['custo_tonelada_evitada']
        valor_credito = analise_tea['receitas']['preco_credito_usado']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Custo por tCO₂eq Evitada",
                f"R$ {formatar_br(custo_tonelada)}",
                help="Custo de abatimento por tonelada de carbono"
            )
        
        with col2:
            st.metric(
                "Preço de Mercado do Crédito",
                f"R$ {formatar_br(valor_credito)}",
                help="Preço atual do crédito de carbono"
            )
        
        with col3:
            diferenca = valor_credito - custo_tonelada
            
            if diferenca > 0:
                delta_color = "normal"
            elif diferenca < 0:
                delta_color = "inverse"
            else:
                delta_color = "off"
            
            if valor_credito > 0 and diferenca != 0:
                delta_percent = (diferenca / valor_credito) * 100
                delta_text = f"{delta_percent:.1f}%"
            else:
                delta_text = None
            
            st.metric(
                "Margem por Crédito",
                f"R$ {formatar_br(diferenca)}",
                delta=delta_text,
                delta_color=delta_color
            )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tecnologias = ['Vermicompostagem', 'Compostagem Tradicional', 'Aterro Energético', 'Incinerador']
        
        custos_ton = [custo_tonelada, custo_tonelada*1.5, custo_tonelada*0.8, custo_tonelada*2.0]
        eficiencia = [90, 70, 50, 85]
        
        scatter = ax.scatter(custos_ton, eficiencia, s=200, 
                           c=['blue', 'orange', 'green', 'red'], 
                           alpha=0.7, edgecolors='black')
        
        for i, tech in enumerate(tecnologias):
            ax.annotate(tech, (custos_ton[i], eficiencia[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)
        
        ax.set_xlabel('Custo por tCO₂eq Evitada (R$)')
        ax.set_ylabel('Eficiência de Redução (%)')
        ax.set_title('Trade-off: Custo vs Eficiência de Diferentes Tecnologias')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig)

def analise_financeira_risco(resultados_mc, preco_carbono, taxa_cambio, nome_metodologia):
    resultados_array = np.array(resultados_mc)
    
    media = np.mean(resultados_array)
    mediana = np.median(resultados_array)
    std = np.std(resultados_array)
    
    p5 = np.percentile(resultados_array, 5)
    p25 = np.percentile(resultados_array, 25)
    p75 = np.percentile(resultados_array, 75)
    p95 = np.percentile(resultados_array, 95)
    
    ic_inferior = np.percentile(resultados_array, 2.5)
    ic_superior = np.percentile(resultados_array, 97.5)
    
    var_95 = np.percentile(resultados_array, 5)
    cvar_95 = resultados_array[resultados_array <= var_95].mean()
    
    valor_medio_eur = media * preco_carbono
    valor_var_eur = var_95 * preco_carbono
    valor_cvar_eur = cvar_95 * preco_carbono
    
    valor_medio_brl = valor_medio_eur * taxa_cambio
    valor_var_brl = valor_var_eur * taxa_cambio
    valor_cvar_brl = valor_cvar_eur * taxa_cambio
    
    downside = media - ic_inferior
    upside = ic_superior - media
    
    downside_brl = downside * preco_carbono * taxa_cambio
    upside_brl = upside * preco_carbono * taxa_cambio
    
    return {
        'nome': nome_metodologia,
        'estatisticas': {
            'media': media,
            'mediana': mediana,
            'std': std,
            'p5': p5,
            'p25': p25,
            'p75': p75,
            'p95': p95,
            'ic_95_inf': ic_inferior,
            'ic_95_sup': ic_superior,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside': downside,
            'upside': upside
        },
        'financeiro_eur': {
            'valor_medio': valor_medio_eur,
            'valor_var': valor_var_eur,
            'valor_cvar': valor_cvar_eur
        },
        'financeiro_brl': {
            'valor_medio': valor_medio_brl,
            'valor_var': valor_var_brl,
            'valor_cvar': valor_cvar_brl,
            'downside': downside_brl,
            'upside': upside_brl
        }
    }

def criar_dashboard_financeiro(analise_tese, analise_unfccc, preco_carbono, taxa_cambio, results_array_tese, results_array_unfccc):
    st.subheader("💰 Dashboard Financeiro de Risco")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral", 
        "🎯 Análise de Risco", 
        "📈 Comparação", 
        "💡 Recomendações"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {analise_tese['nome']}")
            st.metric(
                "Valor Esperado (R$)", 
                f"R$ {formatar_br(analise_tese['financeiro_brl']['valor_medio'])}"
            )
            
            st.info(f"""
            **Inferior:** R$ {formatar_br(analise_tese['financeiro_brl']['valor_medio'] - analise_tese['financeiro_brl']['downside'])}
            **Superior:** R$ {formatar_br(analise_tese['financeiro_brl']['valor_medio'] + analise_tese['financeiro_brl']['upside'])}
            """)
        
        with col2:
            st.markdown(f"#### {analise_unfccc['nome']}")
            st.metric(
                "Valor Esperado (R$)", 
                f"R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_medio'])}"
            )
            
            st.info(f"""
            **Inferior:** R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_medio'] - analise_unfccc['financeiro_brl']['downside'])}
            **Superior:** R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_medio'] + analise_unfccc['financeiro_brl']['upside'])}
            """)
    
    with tab2:
        st.markdown("#### 🎯 Medidas de Risco Financeiro")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "VaR 95% - Tese (R$)",
                f"R$ {formatar_br(analise_tese['financeiro_brl']['valor_var'])}",
                help="Valor em Risco: máxima perda esperada com 95% de confiança"
            )
        
        with col2:
            st.metric(
                "CVaR 95% - Tese (R$)",
                f"R$ {formatar_br(analise_tese['financeiro_brl']['valor_cvar'])}",
                help="Perda esperada nos piores 5% dos cenários"
            )
        
        with col3:
            st.metric(
                "VaR 95% - UNFCCC (R$)",
                f"R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_var'])}",
                help="Valor em Risco: máxima perda esperada com 95% de confiança"
            )
        
        with col4:
            st.metric(
                "CVaR 95% - UNFCCC (R$)",
                f"R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_cvar'])}",
                help="Perda esperada nos piores 5% dos cenários"
            )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valores_tese_brl = results_array_tese * preco_carbono * taxa_cambio
        valores_unfccc_brl = results_array_unfccc * preco_carbono * taxa_cambio
        
        sns.histplot(valores_tese_brl, kde=True, bins=30, color='skyblue', 
                    label='Tese', alpha=0.6, ax=ax)
        sns.histplot(valores_unfccc_brl, kde=True, bins=30, color='coral', 
                    label='UNFCCC', alpha=0.6, ax=ax)
        
        ax.axvline(analise_tese['financeiro_brl']['valor_var'], color='blue', 
                  linestyle='--', label=f"VaR 95% Tese: R$ {formatar_br(analise_tese['financeiro_brl']['valor_var'])}")
        ax.axvline(analise_unfccc['financeiro_brl']['valor_var'], color='red', 
                  linestyle='--', label=f"VaR 95% UNFCCC: R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_var'])}")
        
        ax.set_title('Distribuição do Valor Financeiro dos Créditos de Carbono')
        ax.set_xlabel('Valor (R$)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig)
    
    with tab3:
        st.markdown("#### 📈 Comparação de Retorno vs Risco")
        
        df_comparativo = pd.DataFrame({
            'Métrica': [
                'Valor Esperado (R$)', 
                'Downside (R$)', 
                'Upside (R$)',
                'VaR 95% (R$)',
                'CVaR 95% (R$)',
                'Razão Retorno/Risco'
            ],
            'Proposta da Tese': [
                formatar_br(analise_tese['financeiro_brl']['valor_medio']),
                formatar_br(analise_tese['financeiro_brl']['downside']),
                formatar_br(analise_tese['financeiro_brl']['upside']),
                formatar_br(analise_tese['financeiro_brl']['valor_var']),
                formatar_br(analise_tese['financeiro_brl']['valor_cvar']),
                formatar_br(analise_tese['financeiro_brl']['valor_medio'] / analise_tese['financeiro_brl']['valor_cvar'] if analise_tese['financeiro_brl']['valor_cvar'] > 0 else '∞')
            ],
            'Cenário UNFCCC': [
                formatar_br(analise_unfccc['financeiro_brl']['valor_medio']),
                formatar_br(analise_unfccc['financeiro_brl']['downside']),
                formatar_br(analise_unfccc['financeiro_brl']['upside']),
                formatar_br(analise_unfccc['financeiro_brl']['valor_var']),
                formatar_br(analise_unfccc['financeiro_brl']['valor_cvar']),
                formatar_br(analise_unfccc['financeiro_brl']['valor_medio'] / analise_unfccc['financeiro_brl']['valor_cvar'] if analise_unfccc['financeiro_brl']['valor_cvar'] > 0 else '∞')
            ]
        })
        
        st.dataframe(df_comparativo, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(
            analise_tese['financeiro_brl']['valor_cvar'],
            analise_tese['financeiro_brl']['valor_medio'],
            s=200, color='blue', label='Proposta da Tese',
            edgecolors='black', linewidth=2
        )
        
        ax.scatter(
            analise_unfccc['financeiro_brl']['valor_cvar'],
            analise_unfccc['financeiro_brl']['valor_medio'],
            s=200, color='red', label='Cenário UNFCCC',
            edgecolors='black', linewidth=2
        )
        
        ax.plot([0, max(analise_tese['financeiro_brl']['valor_cvar'], 
                       analise_unfccc['financeiro_brl']['valor_cvar'])],
                [0, max(analise_tese['financeiro_brl']['valor_medio'],
                       analise_unfccc['financeiro_brl']['valor_medio'])],
                'k--', alpha=0.3, label='Fronteira de Eficiência')
        
        ax.set_xlabel('Risco (CVaR 95% - R$)')
        ax.set_ylabel('Retorno Esperado (R$)')
        ax.set_title('Trade-off Retorno vs Risco')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(br_format))
        ax.yaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig)
    
    with tab4:
        st.markdown("#### 💡 Recomendações Baseadas em Risco")
        
        if analise_tese['financeiro_brl']['valor_medio'] > analise_unfccc['financeiro_brl']['valor_medio']:
            diferenca_valor = analise_tese['financeiro_brl']['valor_medio'] - analise_unfccc['financeiro_brl']['valor_medio']
            st.success(f"✅ **A Tese oferece R$ {formatar_br(diferenca_valor)} a mais em valor esperado**")
        else:
            st.warning("⚠️ **O cenário UNFCCC tem maior valor esperado**")
        
        if analise_tese['financeiro_brl']['valor_cvar'] > analise_unfccc['financeiro_brl']['valor_cvar']:
            st.warning(f"⚠️ **A Tese tem maior risco de cauda (CVaR): R$ {formatar_br(analise_tese['financeiro_brl']['valor_cvar'])} vs R$ {formatar_br(analise_unfccc['financeiro_brl']['valor_cvar'])}**")
        else:
            st.success("✅ **A Tese tem menor risco de cauda**")
        
        st.markdown("""
        **📋 Recomendações de Decisão:**
        
        1. **Para Investidores Conservadores:**
           - Priorize metodologia com menor CVaR
           - Considere o limite inferior do IC 95% como cenário base
           - Exija margem de segurança maior
        
        2. **Para Investidores Agressivos:**
           - Foque no upside potencial
           - Considere o limite superior do IC 95%
           - Avalie a razão retorno/risco
        """)
        
        cenarios = pd.DataFrame({
            'Cenário': ['Otimista', 'Mais Provável', 'Pessimista', 'Catastrófico'],
            'Probabilidade': ['5%', '90%', '5%', '1%'],
            'Tese - Valor (R$)': [
                formatar_br(analise_tese['estatisticas']['p95'] * preco_carbono * taxa_cambio),
                formatar_br(analise_tese['estatisticas']['media'] * preco_carbono * taxa_cambio),
                formatar_br(analise_tese['estatisticas']['p5'] * preco_carbono * taxa_cambio),
                formatar_br(analise_tese['estatisticas']['cvar_95'] * preco_carbono * taxa_cambio)
            ],
            'UNFCCC - Valor (R$)': [
                formatar_br(analise_unfccc['estatisticas']['p95'] * preco_carbono * taxa_cambio),
                formatar_br(analise_unfccc['estatisticas']['media'] * preco_carbono * taxa_cambio),
                formatar_br(analise_unfccc['estatisticas']['p5'] * preco_carbono * taxa_cambio),
                formatar_br(analise_unfccc['estatisticas']['cvar_95'] * preco_carbono * taxa_cambio)
            ]
        })
        
        st.dataframe(cenarios, use_container_width=True)

def simulacao_cenarios(preco_base, cambio_base, media_tese, media_unfccc):
    st.subheader("🌍 Simulação de Cenários de Mercado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Preço do Carbono")
        variacao_preco = st.slider(
            "Variação no Preço (%)", 
            -50, 100, 0, 10
        )
        novo_preco = preco_base * (1 + variacao_preco/100)
        st.metric("Novo Preço", f"€ {formatar_br(novo_preco)}", 
                 delta=f"{variacao_preco}%")
    
    with col2:
        st.markdown("#### Taxa de Câmbio")
        variacao_cambio = st.slider(
            "Variação no Câmbio (%)", 
            -30, 50, 0, 5
        )
        novo_cambio = cambio_base * (1 + variacao_cambio/100)
        st.metric("Novo Câmbio", f"R$ {formatar_br(novo_cambio)}",
                 delta=f"{variacao_cambio}%")
    
    novo_valor_tese = media_tese * novo_preco * novo_cambio
    novo_valor_unfccc = media_unfccc * novo_preco * novo_cambio
    
    st.markdown("#### 📊 Impacto Financeiro dos Cenários")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cenarios = ['Base', 'Otimista', 'Pessimista']
    valores_tese = [
        media_tese * preco_base * cambio_base,
        media_tese * (preco_base * 1.5) * (cambio_base * 1.2),
        media_tese * (preco_base * 0.5) * (cambio_base * 0.8)
    ]
    
    valores_unfccc = [
        media_unfccc * preco_base * cambio_base,
        media_unfccc * (preco_base * 1.5) * (cambio_base * 1.2),
        media_unfccc * (preco_base * 0.5) * (cambio_base * 0.8)
    ]
    
    x = np.arange(len(cenarios))
    ax.bar(x - 0.2, valores_tese, 0.4, label='Tese', color='blue')
    ax.bar(x + 0.2, valores_unfccc, 0.4, label='UNFCCC', color='red')
    
    ax.set_xlabel('Cenário')
    ax.set_ylabel('Valor (R$)')
    ax.set_title('Sensibilidade Financeira a Cenários de Mercado')
    ax.set_xticks(x)
    ax.set_xticklabels(cenarios)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(br_format))
    
    st.pyplot(fig)

if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação completa...'):
        params_base = [umidade, T, DOC]

        ch4_aterro_dia, n2o_aterro_dia = calcular_emissoes_aterro(params_base)
        ch4_vermi_dia, n2o_vermi_dia = calcular_emissoes_vermi(params_base)

        df = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_aterro_dia,
            'N2O_Aterro_kg_dia': n2o_aterro_dia,
            'CH4_Vermi_kg_dia': ch4_vermi_dia,
            'N2O_Vermi_kg_dia': n2o_vermi_dia,
        })

        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi']:
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * (GWP_CH4_20 if 'CH4' in gas else GWP_N2O_20) / 1000

        df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
        df['Total_Vermi_tCO2eq_dia'] = df['CH4_Vermi_tCO2eq'] + df['N2O_Vermi_tCO2eq']

        df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
        df['Total_Vermi_tCO2eq_acum'] = df['Total_Vermi_tCO2eq_dia'].cumsum()
        df['Reducao_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2eq_acum']

        df['Year'] = df['Data'].dt.year
        df_anual_revisado = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
        }).reset_index()

        df_anual_revisado['Emission reductions (t CO₂eq)'] = df_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_anual_revisado['Total_Vermi_tCO2eq_dia']
        df_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()

        df_anual_revisado.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
            'Total_Vermi_tCO2eq_dia': 'Project emissions (t CO₂eq)',
        }, inplace=True)

        ch4_compost_UNFCCC, n2o_compost_UNFCCC = calcular_emissoes_compostagem(
            params_base, dias_simulacao=dias, dias_compostagem=50
        )
        ch4_compost_unfccc_tco2eq = ch4_compost_UNFCCC * GWP_CH4_20 / 1000
        n2o_compost_unfccc_tco2eq = n2o_compost_UNFCCC * GWP_N2O_20 / 1000
        total_compost_unfccc_tco2eq_dia = ch4_compost_unfccc_tco2eq + n2o_compost_unfccc_tco2eq

        df_comp_unfccc_dia = pd.DataFrame({
            'Data': datas,
            'Total_Compost_tCO2eq_dia': total_compost_unfccc_tco2eq_dia
        })
        df_comp_unfccc_dia['Year'] = df_comp_unfccc_dia['Data'].dt.year

        df_comp_anual_revisado = df_comp_unfccc_dia.groupby('Year').agg({
            'Total_Compost_tCO2eq_dia': 'sum'
        }).reset_index()

        df_comp_anual_revisado = pd.merge(df_comp_anual_revisado,
                                          df_anual_revisado[['Year', 'Baseline emissions (t CO₂eq)']],
                                          on='Year', how='left')

        df_comp_anual_revisado['Emission reductions (t CO₂eq)'] = df_comp_anual_revisado['Baseline emissions (t CO₂eq)'] - df_comp_anual_revisado['Total_Compost_tCO2eq_dia']
        df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_comp_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()
        df_comp_anual_revisado.rename(columns={'Total_Compost_tCO2eq_dia': 'Project emissions (t CO₂eq)'}, inplace=True)

        st.header("📈 Resultados da Simulação")
        
        total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
        total_evitado_unfccc = df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'].iloc[-1]
        
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        fonte_cotacao = st.session_state.fonte_cotacao
        
        valor_tese_eur = calcular_valor_creditos(total_evitado_tese, preco_carbono, moeda)
        valor_unfccc_eur = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, moeda)
        
        valor_tese_brl = calcular_valor_creditos(total_evitado_tese, preco_carbono, "R$", taxa_cambio)
        valor_unfccc_brl = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, "R$", taxa_cambio)
        
        st.subheader("💰 Valor Financeiro das Emissões Evitadas")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbono (Euro)", 
                f"{moeda} {preco_carbono:.2f}/tCO₂eq",
                help=f"Fonte: {fonte_cotacao}"
            )
        with col2:
            st.metric(
                "Valor Tese (Euro)", 
                f"{moeda} {formatar_br(valor_tese_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_tese)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Valor UNFCCC (Euro)", 
                f"{moeda} {formatar_br(valor_unfccc_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_unfccc)} tCO₂eq evitadas"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbono (R$)", 
                f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                help="Preço do carbono convertido para Reais"
            )
        with col2:
            st.metric(
                "Valor Tese (R$)", 
                f"R$ {formatar_br(valor_tese_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_tese)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Valor UNFCCC (R$)", 
                f"R$ {formatar_br(valor_unfccc_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_unfccc)} tCO₂eq evitadas"
            )
        
        st.markdown("#### 🌍 Comparação entre Mercados de Carbono")
        
        preco_voluntario_usd = 7.48
        preco_regulado_eur = 85.57
        taxa_cambio_usd = 5.0
        
        preco_voluntario_brl = preco_voluntario_usd * taxa_cambio_usd
        preco_regulado_brl = preco_regulado_eur * taxa_cambio
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            valor_voluntario = total_evitado_tese * preco_voluntario_brl
            st.metric(
                "Mercado Voluntário",
                f"R$ {formatar_br(valor_voluntario)}",
                help=f"Baseado em USD {preco_voluntario_usd}/tCO₂eq"
            )
        
        with col2:
            valor_hibrido = total_evitado_tese * preco_carbono * taxa_cambio
            st.metric(
                "Mercado Atual",
                f"R$ {formatar_br(valor_hibrido)}",
                help=f"Baseado em {moeda} {preco_carbono:.2f}/tCO₂eq"
            )
        
        with col3:
            valor_regulado = total_evitado_tese * preco_regulado_brl
            st.metric(
                "Mercado Regulado (EU ETS)",
                f"R$ {formatar_br(valor_regulado)}",
                help=f"Baseado em €{preco_regulado_eur:.2f}/tCO₂eq"
            )
        
        st.subheader("📊 Resumo das Emissões Evitadas")
        
        media_anual_tese = total_evitado_tese / anos_simulacao
        media_anual_unfccc = total_evitado_unfccc / anos_simulacao
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📋 Metodologia da Tese")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_tese)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_tese)} tCO₂eq/ano",
                help=f"Emissões evitadas por ano em média"
            )

        with col2:
            st.markdown("#### 📋 Metodologia UNFCCC")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_unfccc)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_unfccc)} tCO₂eq/ano",
                help=f"Emissões evitadas por ano em média"
            )

        st.subheader("📊 Comparação Anual das Emissões Evitadas")
        df_evitadas_anual = pd.DataFrame({
            'Year': df_anual_revisado['Year'],
            'Proposta da Tese': df_anual_revisado['Emission reductions (t CO₂eq)'],
            'UNFCCC (2012)': df_comp_anual_revisado['Emission reductions (t CO₂eq)']
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        br_formatter = FuncFormatter(br_format)
        x = np.arange(len(df_evitadas_anual['Year']))
        bar_width = 0.35

        ax.bar(x - bar_width/2, df_evitadas_anual['Proposta da Tese'], width=bar_width,
                label='Proposta da Tese', edgecolor='black')
        ax.bar(x + bar_width/2, df_evitadas_anual['UNFCCC (2012)'], width=bar_width,
                label='UNFCCC (2012)', edgecolor='black', hatch='//')

        for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Proposta da Tese'], 
                                         df_evitadas_anual['UNFCCC (2012)'])):
            ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                    formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
            ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                    formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
        ax.set_title('Comparação Anual das Emissões Evitadas: Proposta da Tese vs UNFCCC (2012)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual_revisado['Year'], fontsize=8)

        ax.legend(title='Metodologia')
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.subheader("📉 Redução de Emissões Acumulada")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], 'r-', label='Cenário Base (Aterro Sanitário)', linewidth=2)
        ax.plot(df['Data'], df['Total_Vermi_tCO2eq_acum'], 'g-', label='Projeto (Compostagem em reatores com minhocas)', linewidth=2)
        ax.fill_between(df['Data'], df['Total_Vermi_tCO2eq_acum'], df['Total_Aterro_tCO2eq_acum'],
                        color='skyblue', alpha=0.5, label='Emissões Evitadas')
        ax.set_title('Redução de Emissões em {} Anos'.format(anos_simulacao))
        ax.set_xlabel('Ano')
        ax.set_ylabel('tCO₂eq Acumulado')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(br_formatter)

        st.pyplot(fig)

        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Proposta da Tese")
        br_formatter_sobol = FuncFormatter(br_format)

        np.random.seed(50)  
        
        problem_tese = {
            'num_vars': 3,
            'names': ['umidade', 'T', 'DOC'],
            'bounds': [
                [0.5, 0.85],
                [25.0, 45.0],
                [0.15, 0.50],
            ]
        }

        param_values_tese = sample(problem_tese, n_samples)
        results_tese = Parallel(n_jobs=-1)(delayed(executar_simulacao_completa)(params) for params in param_values_tese)
        Si_tese = analyze(problem_tese, np.array(results_tese), print_to_console=False)
        
        sensibilidade_df_tese = pd.DataFrame({
            'Parámetro': problem_tese['names'],
            'S1': Si_tese['S1'],
            'ST': Si_tese['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parámetro', data=sensibilidade_df_tese, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Proposta da Tese')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        st.pyplot(fig)

        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Cenário UNFCCC")

        np.random.seed(50)
        
        problem_unfccc = {
            'num_vars': 3,
            'names': ['umidade', 'T', 'DOC'],
            'bounds': [
                [0.5, 0.85],
                [25, 45],
                [0.15, 0.50],
            ]
        }

        param_values_unfccc = sample(problem_unfccc, n_samples)
        results_unfccc = Parallel(n_jobs=-1)(delayed(executar_simulacao_unfccc)(params) for params in param_values_unfccc)
        Si_unfccc = analyze(problem_unfccc, np.array(results_unfccc), print_to_console=False)
        
        sensibilidade_df_unfccc = pd.DataFrame({
            'Parámetro': problem_unfccc['names'],
            'S1': Si_unfccc['S1'],
            'ST': Si_unfccc['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parámetro', data=sensibilidade_df_unfccc, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Cenário UNFCCC')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        st.pyplot(fig)

        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Proposta da Tese")
        
        def gerar_parametros_mc_tese(n):
            np.random.seed(50)
            umidade_vals = np.random.uniform(0.75, 0.90, n)
            temp_vals = np.random.normal(25, 3, n)
            doc_vals = np.random.triangular(0.12, 0.15, 0.18, n)
            
            return umidade_vals, temp_vals, doc_vals

        umidade_vals, temp_vals, doc_vals = gerar_parametros_mc_tese(n_simulations)
        
        results_mc_tese = []
        for i in range(n_simulations):
            params_tese = [umidade_vals[i], temp_vals[i], doc_vals[i]]
            results_mc_tese.append(executar_simulacao_completa(params_tese))

        results_array_tese = np.array(results_mc_tese)
        media_tese = np.mean(results_array_tese)
        intervalo_95_tese = np.percentile(results_array_tese, [2.5, 97.5])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_tese, kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(media_tese, color='red', linestyle='--', label=f'Média: {formatar_br(media_tese)} tCO₂eq')
        ax.axvline(intervalo_95_tese[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_tese[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Proposta da Tese')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Cenário UNFCCC")
        
        def gerar_parametros_mc_unfccc(n):
            np.random.seed(50)
            umidade_vals = np.random.uniform(0.75, 0.90, n)
            temp_vals = np.random.normal(25, 3, n)
            doc_vals = np.random.triangular(0.12, 0.15, 0.18, n)
            
            return umidade_vals, temp_vals, doc_vals

        umidade_vals, temp_vals, doc_vals = gerar_parametros_mc_unfccc(n_simulations)
        
        results_mc_unfccc = []
        for i in range(n_simulations):
            params_unfccc = [umidade_vals[i], temp_vals[i], doc_vals[i]]
            results_mc_unfccc.append(executar_simulacao_unfccc(params_unfccc))

        results_array_unfccc = np.array(results_mc_unfccc)
        media_unfccc = np.mean(results_array_unfccc)
        intervalo_95_unfccc = np.percentile(results_array_unfccc, [2.5, 97.5])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_unfccc, kde=True, bins=30, color='coral', ax=ax)
        ax.axvline(media_unfccc, color='red', linestyle='--', label=f'Média: {formatar_br(media_unfccc)} tCO₂eq')
        ax.axvline(intervalo_95_unfccc[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_unfccc[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Cenário UNFCCC')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        st.header("💰 Análise Financeira de Risco Detalhada")

        analise_tese = analise_financeira_risco(
            resultados_mc=results_mc_tese,
            preco_carbono=st.session_state.preco_carbono,
            taxa_cambio=st.session_state.taxa_cambio,
            nome_metodologia="Proposta da Tese"
        )

        analise_unfccc = analise_financeira_risco(
            resultados_mc=results_mc_unfccc,
            preco_carbono=st.session_state.preco_carbono,
            taxa_cambio=st.session_state.taxa_cambio,
            nome_metodologia="Cenário UNFCCC"
        )

        criar_dashboard_financeiro(
            analise_tese=analise_tese,
            analise_unfccc=analise_unfccc,
            preco_carbono=st.session_state.preco_carbono,
            taxa_cambio=st.session_state.taxa_cambio,
            results_array_tese=results_array_tese,
            results_array_unfccc=results_array_unfccc
        )

        st.markdown("---")
        st.header("🏭 Análise Técnico-Econômica Integrada")
        
        parametros_tea = {
            'fator_capex': st.session_state.get('fator_capex', 1.0),
            'fator_opex': st.session_state.get('fator_opex', 1.0),
            'mercado_carbono': st.session_state.get('mercado_carbono', "Híbrido (Média)"),
            'preco_humus': st.session_state.get('preco_humus', 2.5),
            'taxa_desconto': st.session_state.get('taxa_desconto', 0.08),
            'custo_aterro': st.session_state.get('custo_aterro', 0.15) if 'custo_aterro' in st.session_state else 0.15
        }
        
        reducao_anual_tese = media_anual_tese
        reducao_anual_unfccc = media_anual_unfccc
        
        custos_tese = calcular_custos_capex_opex(residuos_kg_dia, anos_simulacao)
        
        custos_tese['capex_total'] *= parametros_tea['fator_capex']
        custos_tese['opex_anual'] *= parametros_tea['fator_opex']
        
        mercado_selecionado = parametros_tea['mercado_carbono']
        if mercado_selecionado == "Voluntário (USD 7.48)":
            preco_carbono_tea = 37.40
        elif mercado_selecionado == "Regulado (EU ETS €85.57)":
            preco_carbono_tea = 544.23
        elif mercado_selecionado == "Customizado":
            preco_carbono_tea = st.session_state.get('preco_carbono_custom', 290.82)
        else:
            preco_carbono_tea = 290.82
        
        receitas_tese = calcular_receitas(
            residuos_kg_dia, 
            reducao_anual_tese,
            preco_carbono_tea,
            mercado='regulado' if preco_carbono_tea > 500 else 'voluntario',
            preco_humus=parametros_tea['preco_humus'],
            custo_aterro=parametros_tea['custo_aterro']
        )
        
        indicadores_tese = calcular_indicadores_financeiros(
            custos_tese['capex_total'],
            custos_tese['opex_anual'],
            receitas_tese['receita_total_anual'],
            anos_simulacao,
            parametros_tea['taxa_desconto']
        )
        
        sensibilidade_tese = analise_sensibilidade_tea(
            residuos_kg_dia, 
            reducao_anual_tese, 
            anos_simulacao,
            preco_humus=parametros_tea['preco_humus'],
            custo_aterro=parametros_tea['custo_aterro']
        )
        
        analise_tea_completa = {
            'capex_total': custos_tese['capex_total'],
            'opex_anual': custos_tese['opex_anual'],
            'capex_detalhado': custos_tese['capex_detalhado'],
            'opex_detalhado': custos_tese['opex_detalhado'],
            'receitas': receitas_tese,
            'indicadores': indicadores_tese
        }
        
        criar_dashboard_tea(analise_tea_completa, sensibilidade_tese)
        
        with st.expander("📋 Resumo Executivo TEA", expanded=True):
            st.markdown(f"""
            ## 📊 Resumo Executivo - Análise Técnico-Econômica
            
            **💼 Viabilidade Financeira:**
            - **VPL:** R$ {formatar_br(indicadores_tese['vpl'])} 
            - **TIR:** {indicadores_tese['tir']*100 if indicadores_tese['tir'] else 'N/A':.1f}%
            - **Payback:** {indicadores_tese['payback_anos'] or '> período'} anos
            - **Custo por tCO₂eq evitada:** R$ {formatar_br(indicadores_tese['custo_tonelada_evitada'])}
            
            **💰 Estrutura de Custos e Receitas:**
            - **Investimento (CAPEX):** R$ {formatar_br(custos_tese['capex_total'])}
            - **Custo Anual (OPEX):** R$ {formatar_br(custos_tese['opex_anual'])}/ano
            - **Receita Total Anual:** R$ {formatar_br(receitas_tese['receita_total_anual'])}/ano
              - Créditos de Carbono: R$ {formatar_br(receitas_tese['receita_carbono'])}/ano
              - Venda de Húmus: R$ {formatar_br(receitas_tese['receita_humus'])}/ano
              - Economia com Aterro: R$ {formatar_br(receitas_tese['economia_aterro'])}/ano
            
            **🌍 Impacto Econômico-Ambiental:**
            - **Custo de Abatimento:** R$ {formatar_br(indicadores_tese['custo_tonelada_evitada'])}/tCO₂eq
            - **Preço de Mercado:** R$ {formatar_br(preco_carbono_tea)}/tCO₂eq
            - **Margem por Crédito:** R$ {formatar_br(preco_carbono_tea - indicadores_tese['custo_tonelada_evitada'])}
            - **Produção de Húmus:** {formatar_br(receitas_tese['producao_humus'])} kg/ano
            
            **🎯 Cenários de Mercado:**
            - **Voluntário (USD 7.48):** VPL = R$ {formatar_br(sensibilidade_tese['Pessimista']['indicadores']['vpl'])}
            - **Híbrido (Média):** VPL = R$ {formatar_br(sensibilidade_tese['Base']['indicadores']['vpl'])}
            - **Regulado (EU ETS):** VPL = R$ {formatar_br(sensibilidade_tese['Otimista']['indicadores']['vpl'])}
            
            **⚖️ Conclusão TEA:**
            {"✅ **PROJETO VIÁVEL** - VPL positivo e TIR acima do custo de capital" 
             if indicadores_tese['vpl'] > 0 else 
             "⚠️ **PROJETO NÃO VIÁVEL** - Necessita de ajustes ou incentivos"}
            """)

        simulacao_cenarios(
            preco_base=st.session_state.preco_carbono,
            cambio_base=st.session_state.taxa_cambio,
            media_tese=media_tese,
            media_unfccc=media_unfccc
        )

        st.subheader("📊 Análise Estatística de Comparação")
        
        diferencas = results_array_tese - results_array_unfccc
        _, p_valor_normalidade_diff = stats.normaltest(diferencas)
        st.write(f"Teste de normalidade das diferenças (p-value): **{p_valor_normalidade_diff:.5f}**")

        ttest_pareado, p_ttest_pareado = stats.ttest_rel(results_array_tese, results_array_unfccc)
        st.write(f"Teste T pareado: Estatística t = **{ttest_pareado:.5f}**, P-valor = **{p_ttest_pareado:.5f}**")

        wilcoxon_stat, p_wilcoxon = stats.wilcoxon(results_array_tese, results_array_unfccc)
        st.write(f"Teste de Wilcoxon (pareado): Estatística = **{wilcoxon_stat:.5f}**, P-valor = **{p_wilcoxon:.5f}**")

        st.subheader("📋 Resultados Anuais - Proposta da Tese")

        df_anual_formatado = df_anual_revisado.copy()
        for col in df_anual_formatado.columns:
            if col != 'Year':
                df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)

        st.dataframe(df_anual_formatado)

        st.subheader("📋 Resultados Anuais - Metodologia UNFCCC")

        df_comp_formatado = df_comp_anual_revisado.copy()
        for col in df_comp_formatado.columns:
            if col != 'Year':
                df_comp_formatado[col] = df_comp_formatado[col].apply(formatar_br)

        st.dataframe(df_comp_formatado)

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação Completa' para ver os resultados.")

st.markdown("---")
st.markdown("""
**📚 Referências por Cenário:**

**Cenário de Baseline (Aterro Sanitário):**
- Metano: IPCC (2006), UNFCCC (2016) e Wang et al. (2023) 
- Óxido Nitroso: Wang et al. (2017)
- Metano e Óxido Nitroso no pré-descarte: Feng et al. (2020)

**Proposta da Tese (Compostagem em reatores com minhocas):**
- Metano e Óxido Nitroso: Yang et al. (2017)

**Cenário UNFCCC (Compostagem sem minhocas a céu aberto):**
- Protocolo AMS-III.F: UNFCCC (2016)
- Fatores de emissões: Yang et al. (2017)

**🌍 Mercados de Carbono:**
- **Mercado Voluntário:** State of Voluntary Carbon Markets 2024 (USD 7.48/tCO₂eq)
- **Mercado Regulado:** EU ETS Futures Dec/2025 (€85.57/tCO₂eq)
- **Câmbio:** Taxas de referência BCB e mercado
""")
