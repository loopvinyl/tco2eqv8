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

np.random.seed(50)  # Garante reprodutibilidade

# Configurações iniciais
st.set_page_config(page_title="Simulador de Emissões CO₂eq - Brasil", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# FUNÇÕES DE COTAÇÃO AUTOMÁTICA DO CARBONO E CÂMBIO - BRASIL
# =============================================================================

def obter_cotacao_carbono_investing():
    """
    Obtém a cotação em tempo real do carbono via web scraping do Investing.com
    """
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
        
        # Várias estratégias para encontrar o preço
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
                    # Remover caracteres não numéricos exceto ponto
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        # Tentativa alternativa: procurar por padrões numéricos no HTML
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
                    if 50 < preco < 200:  # Faixa razoável para carbono
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except ValueError:
                    continue
                    
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    """
    Obtém a cotação em tempo real do carbono - usa apenas Investing.com
    """
    # Tentar via Investing.com
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    # Fallback para valor padrão (EU ETS Dez/2025)
    return 85.57, "€", "Carbon Emissions (EU ETS Reference)", False, "EU ETS Reference Price"

def obter_cotacao_euro_real():
    """
    Obtém a cotação em tempo real do Euro em relação ao Real Brasileiro
    """
    try:
        # API do BCB
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        # Fallback para API alternativa
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    # Fallback para valor de referência (taxa usada para conversão EU ETS)
    return 6.36, "R$", False, "Reference Rate for EU ETS"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    """
    Calcula o valor financeiro das emissões evitadas baseado no preço do carbono
    """
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    """
    Exibe a cotação do carbono com informações - ATUALIZADA AUTOMATICAMENTE
    """
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    # Atualização automática na primeira execução
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    
    # Botão para atualizar cotações
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    
    # Mostrar mensagem de atualização se necessário
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        
        # Obter cotação do carbono
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        
        # Obter cotação do Euro
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        # Atualizar session state
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        # Resetar flags
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    # Exibe cotação atual do carbono
    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    # Exibe cotação atual do Euro
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {st.session_state.taxa_cambio:.2f}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    # Calcular preço do carbono em Reais
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {preco_carbono_reais:.2f}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    # Informações adicionais com comparação de mercados
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        # Preços de referência dos diferentes mercados
        preco_voluntario_usd = 7.48
        preco_regulado_eur = 85.57
        taxa_cambio_usd = 5.0  # USD/BRL estimado
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
        
        **📈 Mercado de Referência:**
        - European Union Allowances (EUA)
        - European Emissions Trading System (EU ETS)
        - Contratos futuros de carbono (Dec/2025: €85.57)
        - Preços em tempo real
        
        **🔄 Atualização:**
        - As cotações são carregadas automaticamente ao abrir o aplicativo
        - Clique em **"Atualizar Cotações"** para obter valores mais recentes
        - Em caso de falha na conexão, são utilizados valores de referência atualizados
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
        - Análise TEA inclui cenários com diferentes mercados
        """)

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

# Inicializar todas as variáveis de session state necessárias
def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        # Buscar cotação automaticamente na inicialização
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        # Buscar cotação do Euro automaticamente
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

# Chamar a inicialização
inicializar_session_state()

# =============================================================================
# FUNÇÕES ORIGINAIS DO SEU SCRIPT
# =============================================================================

# Função para formatar números no padrão brasileiro
def formatar_br(numero):
    """
    Formata números no padrão brasileiro: 1.234,56
    """
    if pd.isna(numero):
        return "N/A"
    
    # Arredonda para 2 casas decimais
    numero = round(numero, 2)
    
    # Formata como string e substitui o ponto pela vírgula
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função de formatação para os gráficos
def br_format(x, pos):
    """
    Função de formatação para eixos de gráficos (padrão brasileiro)
    """
    if x == 0:
        return "0"
    
    # Para valores muito pequenos, usa notação científica
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    
    # Para valores grandes, formata com separador de milhar
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Para valores menores, mostra duas casas decimais
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format_5_dec(x, pos):
    """
    Função de formatação para eixos de gráficos (padrão brasileiro com 5 decimais)
    """
    return f"{x:,.5f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Título do aplicativo
st.title("🇧🇷 Simulador de Emissões de tCO₂eq - Contexto Brasileiro")
st.markdown("""
**Adaptação de Zziwa et al. (2021) para realidade brasileira**

Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para dois contextos de gestão de resíduos, 
incluindo análise financeira detalhada com valores brasileiros e cenários de mercado.
""")

# =============================================================================
# SIDEBAR COM PARÂMETROS - VALORES BRASILEIROS (AJUSTADOS CONFORME TABELA 18)
# =============================================================================

# Seção de cotação do carbono - AGORA ATUALIZADA AUTOMATICAMENTE
exibir_cotacao_carbono()

# Seção original de parâmetros
with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada - Brasil")
    
    # Entrada principal de resíduos
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", 
                               min_value=10, max_value=1000, value=100, step=10,
                               help="Quantidade diária de resíduos orgânicos gerados - Escala: 100 kg/dia = 36,5 ton/ano")
    
    st.subheader("📊 Parâmetros Operacionais")
    
    # Umidade com formatação brasileira (0,85 em vez de 0.85)
    umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1,
                             help="Percentual de umidade dos resíduos orgânicos")
    umidade = umidade_valor / 100.0
    st.write(f"**Umidade selecionada:** {formatar_br(umidade_valor)}%")
    
    massa_exposta_kg = st.slider("Massa exposta na frente de trabalho (kg)", 50, 200, 100, 10,
                                help="Massa de resíduos exposta diariamente para tratamento")
    h_exposta = st.slider("Horas expostas por dia", 4, 24, 8, 1,
                         help="Horas diárias de exposição dos resíduos")
    
    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5,
                              help="Período total da simulação em anos")
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50,
                             help="Número de iterações para análise de incerteza")
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16,
                         help="Número de amostras para análise de sensibilidade")
    
    # =============================================================================
    # PARÂMETROS TEA (ANÁLISE TÉCNICO-ECONÔMICA) - BRASIL (AJUSTADOS TABELA 18)
    # =============================================================================
    with st.expander("🏭 Parâmetros TEA - Contexto Brasileiro (Tabela 18)"):
        st.markdown("#### 💼 Parâmetros de Custo - Brasil")
        
        # Fatores de ajuste de custo
        fator_capex = st.slider(
            "Fator de ajuste CAPEX", 
            0.5, 2.0, 1.0, 0.1,
            help="Ajuste os custos de investimento para realidade local",
            key="fator_capex"
        )
        
        fator_opex = st.slider(
            "Fator de ajuste OPEX", 
            0.5, 2.0, 1.0, 0.1,
            help="Ajuste os custos operacionais para realidade local",
            key="fator_opex"
        )
        
        st.markdown("#### 📈 Parâmetros de Mercado - Brasil")
        
        # Seleção de mercado de carbono
        mercado_carbono = st.selectbox(
            "Mercado de Carbono para Análise",
            ["Híbrido (Média R$ 290,82)", "Voluntário (R$ 37,40)", "Regulado EU ETS (R$ 544,23)", "Customizado"],
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
        
        # Preço do húmus - AJUSTADO CONFORME TABELA 18
        preco_humus = st.number_input(
            "Preço do Húmus (R$/kg)",
            min_value=1.0,
            value=10.0,  # Corrigido: R$ 10,00 (era R$ 2,50)
            step=0.5,
            key="preco_humus",
            help="Preço de mercado do húmus orgânico - Tabela 18: R$ 10,00/kg"
        )
        
        # Preço das minhocas - NOVO PARÂMETRO TABELA 18
        preco_minhoca = st.number_input(
            "Preço da Minhoca (R$/kg)",
            min_value=50.0,
            value=100.0,  # Novo: R$ 100,00/kg
            step=5.0,
            key="preco_minhoca",
            help="Preço de mercado da minhoca - Tabela 18: R$ 100,00/kg"
        )
        
        # Taxa de desconto
        taxa_desconto = st.slider(
            "Taxa de desconto para VPL (%)",
            0.0, 20.0, 8.0, 0.5,
            key="taxa_desconto",
            help="Taxa Mínima de Atratividade (TMA) - SELIC + risco"
        ) / 100
        
        # Custos de referência - BRASIL
        st.markdown("#### 📊 Custos de Referência - Brasil")
        custo_aterro = st.number_input(
            "Custo de disposição em aterro (R$/kg)",
            min_value=0.05,
            value=0.30,
            step=0.01,
            help="Custo de descarte em aterro sanitário - R$ 300/tonelada",
            key="custo_aterro"
        )
    
    # Informações sobre valores brasileiros - ATUALIZADO CONFORME TABELA 18
    with st.expander("🇧🇷 Valores de Referência - Brasil (Tabela 18)"):
        st.markdown(f"""
        **💼 Valores da Tabela 18 - Adaptação de Zziwa et al. (2021):**
        
        **Para {residuos_kg_dia} kg/dia ({residuos_kg_dia*365/1000:.1f} ton/ano):**
        
        **🏗️ CAPEX (Investimento):**
        - **Reatores:** {residuos_kg_dia} unidades × R$ 1.000 = R$ {formatar_br(residuos_kg_dia*1000)}
        - **Minhocas iniciais:** {residuos_kg_dia*3} kg × R$ 100 = R$ {formatar_br(residuos_kg_dia*300)}
        - **Investimento total:** R$ {formatar_br(residuos_kg_dia*1000 + residuos_kg_dia*300)}
        
        **💰 OPEX (Operação - Anual):**
        - Mão de obra: 2h/dia × R$ 20/h × 365 dias = R$ 14.600
        - Energia: 0,5 kWh/dia × R$ 0,80/kWh × 365 dias = R$ 146
        - Manutenção: 5% do CAPEX
        - Insumos: R$ 0,10/kg de resíduo tratado
        
        **💵 Receitas (para 100 kg/dia):**
        - **Húmus:** 14.600 kg/ano × R$ 10 = R$ 146.000
        - **Minhocas:** 745 kg/ano × R$ 100 = R$ 74.496
        - **Economia aterro:** 36,5 ton × R$ 300 = R$ 10.950
        - **Receita total sem carbono:** R$ 231.446
        
        **📈 Dados de Produção:**
        - Resíduos processados: {residuos_kg_dia*365/1000:.1f} ton/ano
        - Reatores necessários: {residuos_kg_dia} unidades
        - Minhocas iniciais: {residuos_kg_dia*3} kg
        - Produção anual de húmus: {residuos_kg_dia*365/1000*0.4:.1f} ton (40% dos resíduos)
        - Produção anual de minhocas: {745*(residuos_kg_dia/100):.1f} kg
        """)
    
    if st.button("🚀 Executar Simulação Completa", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# PARÂMETROS FIXOS (DO CÓDIGO ORIGINAL)
# =============================================================================

T = 25  # Temperatura média (ºC) - Brasil
DOC = 0.15  # Carbono orgânico degradável (fração)
DOCf_val = 0.0147 * T + 0.28
MCF = 1  # Fator de correção de metano
F = 0.5  # Fração de metano no biogás
OX = 0.1  # Fator de oxidação
Ri = 0.0  # Metano recuperado

# Constante de decaimento (fixa como no script anexo)
k_ano = 0.06  # Constante de decaimento anual

# Vermicompostagem (Yang et al. 2017) - valores fixos
TOC_YANG = 0.436  # Fração de carbono orgânico total
TN_YANG = 14.2 / 1000  # Fração de nitrogênio total
CH4_C_FRAC_YANG = 0.13 / 100  # Fração do TOC emitida como CH4-C (fixo)
N2O_N_FRAC_YANG = 0.92 / 100  # Fração do TN emitida como N2O-N (fixo)
DIAS_COMPOSTAGEM = 50  # Período total de compostagem

# Perfil temporal de emissões baseado em Yang et al. (2017)
PERFIL_CH4_VERMI = np.array([
    0.02, 0.02, 0.02, 0.03, 0.03,  # Dias 1-5
    0.04, 0.04, 0.05, 0.05, 0.06,  # Dias 6-10
    0.07, 0.08, 0.09, 0.10, 0.09,  # Dias 11-15
    0.08, 0.07, 0.06, 0.05, 0.04,  # Dias 16-20
    0.03, 0.02, 0.02, 0.01, 0.01,  # Dias 21-25
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 36-40
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_CH4_VERMI /= PERFIL_CH4_VERMI.sum()

PERFIL_N2O_VERMI = np.array([
    0.15, 0.10, 0.20, 0.05, 0.03,  # Dias 1-5 (pico no dia 3)
    0.03, 0.03, 0.04, 0.05, 0.06,  # Dias 6-10
    0.08, 0.09, 0.10, 0.08, 0.07,  # Dias 11-15
    0.06, 0.05, 0.04, 0.03, 0.02,  # Dias 16-20
    0.01, 0.01, 0.005, 0.005, 0.005,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_N2O_VERMI /= PERFIL_N2O_VERMI.sum()

# Emissões pré-descarte (Feng et al. 2020)
CH4_pre_descarte_ugC_por_kg_h_min = 0.18
CH4_pre_descarte_ugC_por_kg_h_max = 5.38
CH4_pre_descarte_ugC_por_kg_h_media = 2.78

fator_conversao_C_para_CH4 = 16/12
CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000

N2O_pre_descarte_mgN_por_kg = 20.26
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000

PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}

# GWP (IPCC AR6)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Período de Simulação
dias = anos_simulacao * 365
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)
datas = pd.date_range(start=data_inicio, periods=dias, freq='D')

# Perfil temporal N2O (Wang et al. 2017)
PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}

# Valores específicos para compostagem termofílica (Yang et al. 2017) - valores fixos
CH4_C_FRAC_THERMO = 0.006  # Fixo
N2O_N_FRAC_THERMO = 0.0196  # Fixo

PERFIL_CH4_THERMO = np.array([
    0.01, 0.02, 0.03, 0.05, 0.08,  # Dias 1-5
    0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico termofílico)
    0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
    0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_CH4_THERMO /= PERFIL_CH4_THERMO.sum()

PERFIL_N2O_THERMO = np.array([
    0.10, 0.08, 0.15, 0.05, 0.03,  # Dias 1-5
    0.04, 0.05, 0.07, 0.10, 0.12,  # Dias 6-10
    0.15, 0.18, 0.20, 0.18, 0.15,  # Dias 11-15 (pico termofílico)
    0.12, 0.10, 0.08, 0.06, 0.05,  # Dias 16-20
    0.04, 0.03, 0.02, 0.02, 0.01,  # Dias 21-25
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001,   # Dias 46-50
])
PERFIL_N2O_THERMO /= PERFIL_N2O_THERMO.sum()

# =============================================================================
# FUNÇÕES DE CÁLCULO (ADAPTADAS DO SCRIPT ANEXO)
# =============================================================================

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
    
    # Usando valores fixos para CH4_C_FRAC_YANG e N2O_N_FRAC_YANG
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
    
    # Usando valores fixos para CH4_C_FRAC_THERMO e N2O_N_FRAC_THERMO
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

# =============================================================================
# FUNÇÕES PARA ANÁLISE TÉCNICO-ECONÔMICA (TEA) - BRASIL (AJUSTADAS TABELA 18)
# =============================================================================

def calcular_custos_capex_opex_brasil(residuos_kg_dia, anos_operacao):
    """
    Calcula CAPEX e OPEX baseado na capacidade do sistema - CONTEXTO BRASILEIRO
    Baseado nos valores da Tabela 18 da tese (Zziwa et al., 2021 adaptado)
    """
    # CONVERSÕES
    residuos_ton_dia = residuos_kg_dia / 1000
    residuos_ton_ano = residuos_ton_dia * 365
    
    # CAPEX - CUSTOS DE INVESTIMENTO (R$) - TABELA 18
    # Para 100 kg/dia: 100 reatores e 300 kg de minhocas
    num_reatores = int(residuos_kg_dia)  # 1 reator por kg/dia (Tabela 18)
    custo_reatores = num_reatores * 1000  # R$ 1.000 por reator
    
    # Biomassa de minhocas: 3 kg por kg/dia de resíduo (300 kg para 100 kg/dia)
    kg_minhocas = residuos_kg_dia * 3
    custo_minhocas = kg_minhocas * 100  # R$ 100/kg
    
    # CAPEX TOTAL (Tabela 18) - APENAS REATORES E MINHOCAS
    capex_total = custo_reatores + custo_minhocas
    
    # OPEX - CUSTOS OPERACIONAIS ANUAIS (R$/ano)
    # Mantido do script original para análise financeira completa
    # 1. Mão de obra
    custo_mao_de_obra = 2 * 20 * 365  # R$/ano
    
    # 2. Energia elétrica
    custo_energia = 0.5 * 0.80 * 365  # R$/ano
    
    # 3. Manutenção preventiva e corretiva
    custo_manutencao = capex_total * 0.05  # 5% do CAPEX/ano
    
    # 4. Insumos (substrato, correções, etc.)
    custo_insumos = residuos_kg_dia * 0.10 * 365  # R$ 0,10/kg de resíduo tratado
    
    # 5. Administrativo, impostos e taxas
    custo_administrativo = (custo_mao_de_obra + custo_energia + custo_manutencao + custo_insumos) * 0.1  # 10%
    
    # OPEX TOTAL ANUAL
    opex_anual = (custo_mao_de_obra + custo_energia + custo_manutencao + 
                  custo_insumos + custo_administrativo)
    
    # Detalhamento para relatório
    capex_detalhado = {
        'Reatores de vermicompostagem': custo_reatores,
        'Minhocas (Eisenia fetida)': custo_minhocas
    }
    
    opex_detalhado = {
        'Mão de obra operacional': custo_mao_de_obra,
        'Energia elétrica': custo_energia,
        'Manutenção preventiva/corretiva': custo_manutencao,
        'Insumos (substrato, correções)': custo_insumos,
        'Administrativo, impostos e taxas': custo_administrativo
    }
    
    # Informações adicionais do sistema - TABELA 18
    info_sistema = {
        'num_reatores': num_reatores,
        'kg_minhocas': kg_minhocas,
        'capacidade_tratamento_ton_ano': residuos_ton_ano,
        'custo_disposicao_aterro_ano': residuos_ton_ano * 300,
        'producao_humus_ton_ano': residuos_ton_ano * 0.4,  # 40% conversão (Tabela 18)
        'producao_minhocas_kg_ano': 7.45 * residuos_kg_dia  # 745 kg para 100 kg/dia
    }
    
    return {
        'capex_total': capex_total,
        'opex_anual': opex_anual,
        'capex_detalhado': capex_detalhado,
        'opex_detalhado': opex_detalhado,
        'info_sistema': info_sistema,
        'capex_por_kg_dia': capex_total / residuos_kg_dia if residuos_kg_dia > 0 else 0,
        'opex_por_kg_dia': opex_anual / (residuos_kg_dia * 365) if residuos_kg_dia > 0 else 0
    }

def calcular_receitas_brasil(residuos_kg_dia, reducao_anual_tco2eq, preco_carbono_r, 
                           mercado='hibrido', preco_humus=10.0, preco_minhoca=100.0, 
                           custo_aterro=0.30):
    """
    Calcula receitas anuais do projeto - CONTEXTO BRASILEIRO
    Baseado na Tabela 18 da tese
    """
    # CONVERSÕES
    residuos_ton_ano = (residuos_kg_dia / 1000) * 365
    
    # 1. PRODUÇÃO E VENDA DE HÚMUS (40% dos resíduos - Tabela 18)
    producao_humus_ton_ano = residuos_ton_ano * 0.4
    producao_humus_kg_ano = producao_humus_ton_ano * 1000
    receita_humus = producao_humus_kg_ano * preco_humus  # R$/ano
    
    # 2. PRODUÇÃO E VENDA DE MINHOCAS (7,45 kg por kg/dia de resíduo - Tabela 18)
    producao_minhocas_kg_ano = 7.45 * residuos_kg_dia
    receita_minhocas = producao_minhocas_kg_ano * preco_minhoca  # R$/ano
    
    # 3. RECEITA COM CRÉDITOS DE CARBONO
    receita_carbono = reducao_anual_tco2eq * preco_carbono_r
    
    # 4. ECONOMIA COM DISPOSIÇÃO EM ATERRO
    economia_aterro = residuos_ton_ano * custo_aterro * 1000  # R$/ano
    
    # 5. RECEITAS DIRETAS (sem benefícios indiretos)
    receitas_diretas = receita_humus + receita_minhocas + receita_carbono + economia_aterro
    
    # 6. BENEFÍCIOS INDIRETOS (10% das receitas diretas)
    beneficios_indiretos = receitas_diretas * 0.1
    
    # RECEITA TOTAL ANUAL
    receita_total_anual = receitas_diretas + beneficios_indiretos
    
    # Estrutura de receitas (percentual)
    if receita_total_anual > 0:
        perc_humus = (receita_humus / receita_total_anual) * 100
        perc_minhocas = (receita_minhocas / receita_total_anual) * 100
        perc_carbono = (receita_carbono / receita_total_anual) * 100
        perc_economia = (economia_aterro / receita_total_anual) * 100
        perc_indiretos = (beneficios_indiretos / receita_total_anual) * 100
    else:
        perc_humus = perc_minhocas = perc_carbono = perc_economia = perc_indiretos = 0
    
    return {
        'receita_total_anual': receita_total_anual,
        'receita_humus': receita_humus,
        'receita_minhocas': receita_minhocas,
        'receita_carbono': receita_carbono,
        'economia_aterro': economia_aterro,
        'beneficios_indiretos': beneficios_indiretos,
        'producao_humus_kg_ano': producao_humus_kg_ano,
        'producao_minhocas_kg_ano': producao_minhocas_kg_ano,
        'preco_credito_usado': preco_carbono_r,
        'mercado_selecionado': mercado,
        'estrutura_receitas': {
            'humus_perc': perc_humus,
            'minhocas_perc': perc_minhocas,
            'carbono_perc': perc_carbono,
            'economia_aterro_perc': perc_economia,
            'beneficios_indiretos_perc': perc_indiretos
        },
        'parametros_entrada': {
            'residuos_kg_dia': residuos_kg_dia,
            'residuos_ton_ano': residuos_ton_ano,
            'reducao_anual_tco2eq': reducao_anual_tco2eq,
            'preco_humus': preco_humus,
            'preco_minhoca': preco_minhoca,
            'custo_aterro_por_kg': custo_aterro
        }
    }

def calcular_indicadores_financeiros(capex, opex_anual, receita_anual, anos, taxa_desconto=0.08):
    """
    Calcula indicadores financeiros do projeto
    """
    # Fluxo de caixa anual
    fluxo_caixa = [-capex]  # Ano 0
    for ano in range(1, anos + 1):
        fluxo_anual = receita_anual - opex_anual
        fluxo_caixa.append(fluxo_anual)
    
    # VPL (Valor Presente Líquido)
    vpl = 0
    for t, fc in enumerate(fluxo_caixa):
        vpl += fc / ((1 + taxa_desconto) ** t)
    
    # TIR (Taxa Interna de Retorno)
    try:
        tir = np.irr(fluxo_caixa)
    except:
        tir = None
    
    # Payback simples
    acumulado = 0
    payback_anos = None
    for t, fc in enumerate(fluxo_caixa):
        if t == 0:
            continue
        acumulado += fc
        if acumulado >= capex and payback_anos is None:
            payback_anos = t
    
    # Payback descontado
    acumulado_desc = 0
    payback_desc_anos = None
    for t, fc in enumerate(fluxo_caixa):
        if t == 0:
            continue
        fc_desc = fc / ((1 + taxa_desconto) ** t)
        acumulado_desc += fc_desc
        if acumulado_desc >= capex and payback_desc_anos is None:
            payback_desc_anos = t
    
    # Custo por tonelada evitada
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

def analise_sensibilidade_tea_brasil(residuos_kg_dia, reducao_anual_tco2eq, 
                                   anos_simulacao, preco_humus=10.0, preco_minhoca=100.0,
                                   custo_aterro=0.30):
    """
    Realiza análise de sensibilidade dos parâmetros econômicos - CONTEXTO BRASILEIRO
    Baseado na Tabela 18 da tese
    """
    # Parâmetros base (contexto brasileiro)
    custos = calcular_custos_capex_opex_brasil(residuos_kg_dia, anos_simulacao)
    
    # Cenários de sensibilidade específicos para Brasil
    cenarios = {
        'Otimista (Regulado EU ETS)': {
            'capex_fator': 0.90,      # -10% (economia de escala)
            'opex_fator': 0.85,       # -15% (eficiência operacional)
            'receita_fator': 1.30,    # +30% (alto preço carbono)
            'preco_carbono': 544.23,  # Mercado regulado EU ETS (€85.57 * 6,36)
            'preco_humus_fator': 1.25, # +25% (mercado premium)
            'preco_minhoca_fator': 1.25, # +25% (mercado premium)
            'custo_aterro_fator': 1.15 # +15% (aumento taxa aterro)
        },
        'Realista (Híbrido)': {
            'capex_fator': 1.0,
            'opex_fator': 1.0,
            'receita_fator': 1.0,
            'preco_carbono': 290.82,  # Média ponderada
            'preco_humus_fator': 1.0,
            'preco_minhoca_fator': 1.0,
            'custo_aterro_fator': 1.0
        },
        'Pessimista (Voluntário)': {
            'capex_fator': 1.20,      # +20% (custos importação)
            'opex_fator': 1.15,       # +15% (inflação)
            'receita_fator': 0.80,    # -20% (baixo preço carbono)
            'preco_carbono': 37.40,   # Mercado voluntário (USD 7.48 * 5,0)
            'preco_humus_fator': 0.80, # -20% (concorrência)
            'preco_minhoca_fator': 0.80, # -20% (concorrência)
            'custo_aterro_fator': 0.85 # -15% (subsídios)
        },
        'Crítico (Mínimo)': {
            'capex_fator': 1.35,      # +35% (crise econômica)
            'opex_fator': 1.25,       # +25% (alta inflação)
            'receita_fator': 0.65,    # -35% (mercado deprimido)
            'preco_carbono': 18.70,   # Metade do voluntário
            'preco_humus_fator': 0.60, # -40% (mercado saturado)
            'preco_minhoca_fator': 0.60, # -40% (mercado saturado)
            'custo_aterro_fator': 0.70 # -30% (políticas públicas)
        }
    }
    
    resultados = {}
    for cenario, params in cenarios.items():
        capex_ajustado = custos['capex_total'] * params['capex_fator']
        opex_ajustado = custos['opex_anual'] * params['opex_fator']
        
        # Ajustar preços para realidade brasileira
        preco_humus_ajustado = preco_humus * params['preco_humus_fator']
        preco_minhoca_ajustado = preco_minhoca * params['preco_minhoca_fator']
        custo_aterro_ajustado = custo_aterro * params['custo_aterro_fator']
        
        # Calcular receitas ajustadas
        receitas_ajustadas = calcular_receitas_brasil(
            residuos_kg_dia, 
            reducao_anual_tco2eq,
            params['preco_carbono'],
            mercado='regulado' if 'Regulado' in cenario else 'voluntario',
            preco_humus=preco_humus_ajustado,
            preco_minhoca=preco_minhoca_ajustado,
            custo_aterro=custo_aterro_ajustado
        )
        
        receita_ajustada = receitas_ajustadas['receita_total_anual'] * params['receita_fator']
        
        indicadores = calcular_indicadores_financeiros(
            capex_ajustado, 
            opex_ajustado, 
            receita_ajustada,
            anos_simulacao,
            taxa_desconto=0.08  # 8% a.a. (SELIC + risco)
        )
        
        resultados[cenario] = {
            'capex': capex_ajustado,
            'opex_anual': opex_ajustado,
            'receita_anual': receita_ajustada,
            'indicadores': indicadores,
            'receitas_detalhadas': receitas_ajustadas,
            'custos_detalhados': custos,
            'margem_contribuicao': (receita_ajustada - opex_ajustado) / receita_ajustada * 100 if receita_ajustada > 0 else 0
        }
    
    return resultados

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO - COM ADAPTAÇÕES PARA BRASIL (TABELA 18)
# =============================================================================

# Executar simulação quando solicitado
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação completa para contexto brasileiro (Tabela 18)...'):
        # Executar modelo base
        params_base = [umidade, T, DOC]

        ch4_aterro_dia, n2o_aterro_dia = calcular_emissoes_aterro(params_base)
        ch4_vermi_dia, n2o_vermi_dia = calcular_emissoes_vermi(params_base)

        # Construir DataFrame
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

        # Resumo anual
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

        # Cenário UNFCCC
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

        # =============================================================================
        # EXIBIÇÃO DOS RESULTADOS COM COTAÇÃO DO CARBONO E REAL (TABELA 18)
        # =============================================================================

        # Exibir resultados
        st.header("📈 Resultados da Simulação - Brasil (Tabela 18)")
        
        # Obter valores totais
        total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
        total_evitado_unfccc = df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'].iloc[-1]
        
        # Obter preço do carbono e taxa de câmbio da session state
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        fonte_cotacao = st.session_state.fonte_cotacao
        
        # Calcular valores financeiros em Euros
        valor_tese_eur = calcular_valor_creditos(total_evitado_tese, preco_carbono, moeda)
        valor_unfccc_eur = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, moeda)
        
        # Calcular valores financeiros em Reais
        valor_tese_brl = calcular_valor_creditos(total_evitado_tese, preco_carbono, "R$", taxa_cambio)
        valor_unfccc_brl = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, "R$", taxa_cambio)
        
        # NOVA SEÇÃO: VALOR FINANCEIRO DAS EMISSÕES EVITADAS
        st.subheader("💰 Valor Financeiro das Emissões Evitadas")
        
        # Primeira linha: Euros
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
        
        # Segunda linha: Reais
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
        
        # Comparação entre mercados
        st.markdown("#### 🌍 Comparação entre Mercados de Carbono - Brasil (Tabela 19)")
        
        # Preços de referência adaptados para Brasil (Tabela 19)
        preco_voluntario_usd = 7.45  # Tabela 19
        preco_regulado_eur = 72.29   # Tabela 19
        taxa_cambio_usd = 5.65       # Tabela 19 (maio/2025)
        taxa_cambio_eur_t19 = 6.38   # Tabela 19 (maio/2025)
        
        preco_voluntario_brl = preco_voluntario_usd * taxa_cambio_usd
        preco_regulado_brl = preco_regulado_eur * taxa_cambio_eur_t19
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            valor_voluntario = total_evitado_tese * preco_voluntario_brl
            st.metric(
                "Mercado Voluntário (T19)",
                f"R$ {formatar_br(valor_voluntario)}",
                help=f"Baseado em USD {preco_voluntario_usd}/tCO₂eq (R$ {preco_voluntario_brl:.2f}/tCO₂eq) - Tabela 19"
            )
        
        with col2:
            valor_hibrido = total_evitado_tese * preco_carbono * taxa_cambio
            st.metric(
                "Mercado Atual",
                f"R$ {formatar_br(valor_hibrido)}",
                help=f"Baseado em {moeda} {preco_carbono:.2f}/tCO₂eq (R$ {preco_carbono*taxa_cambio:.2f}/tCO₂eq)"
            )
        
        with col3:
            valor_regulado = total_evitado_tese * preco_regulado_brl
            st.metric(
                "Mercado Regulado (T19)",
                f"R$ {formatar_br(valor_regulado)}",
                help=f"Baseado em €{preco_regulado_eur:.2f}/tCO₂eq (R$ {preco_regulado_brl:.2f}/tCO₂eq) - Tabela 19"
            )
        
        # Explicação sobre compra e venda
        with st.expander("💡 Como funciona a comercialização no mercado de carbono - Brasil (Tabelas 18-19)?"):
            st.markdown(f"""
            **📊 Informações de Mercado - Brasil (Tabelas 18-19):**
            - **Preço em Euro:** {moeda} {preco_carbono:.2f}/tCO₂eq
            - **Preço em Real:** R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
            - **Taxa de câmbio atual:** 1 Euro = R$ {taxa_cambio:.2f}
            - **Taxa de câmbio T19 (maio/2025):** 1 Euro = R$ {taxa_cambio_eur_t19:.2f}
            - **Fonte:** {fonte_cotacao}
            
            **🌍 Comparação de Mercados para o Brasil (Tabela 19):**
            - **Mercado Voluntário:** USD {preco_voluntario_usd:.2f} ≈ R$ {preco_voluntario_brl:.2f}/tCO₂eq
            - **Mercado Regulado (EU ETS):** €{preco_regulado_eur:.2f} ≈ R$ {preco_regulado_brl:.2f}/tCO₂eq
            - **Diferença:** {preco_regulado_brl/preco_voluntario_brl:.1f}x maior no regulado
            
            **💰 Valores da Tabela 18 (Receitas Sem Carbono):**
            - Húmus: R$ 146.000 (14.600 kg × R$ 10,00)
            - Minhocas: R$ 74.496 (745 kg × R$ 100,00)
            - Economia aterro: R$ 10.950 (36,5 ton × R$ 300)
            - **Total sem carbono:** R$ 231.446
            
            **💶 Comprar créditos (compensação no Brasil):**
            - Custo em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
            - Custo em Real: **R$ {formatar_br(valor_tese_brl)}**
            
            **💵 Vender créditos (comercialização no Brasil):**  
            - Receita em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
            - Receita em Real: **R$ {formatar_br(valor_tese_brl)}**
            
            **🇧🇷 Mercado Brasileiro Emergente:**
            - Regulamentação em desenvolvimento
            - Potencial para mercado regulado nacional
            - Oportunidades para projetos de compensação
            - Integração com mercados internacionais
            """)
        
        # =============================================================================
        # RESUMO DAS EMISSÕES EVITADAS COM MÉTRICAS ANUAIS
        # =============================================================================
        
        # Métricas de emissões evitadas - layout reorganizado
        st.subheader("📊 Resumo das Emissões Evitadas")
        
        # Calcular médias anuais
        media_anual_tese = total_evitado_tese / anos_simulacao
        media_anual_unfccc = total_evitado_unfccc / anos_simulacao
        
        # Layout com duas colunas principais
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

        # Adicionar explicação sobre as métricas anuais
        with st.expander("💡 Entenda as métricas anuais"):
            st.markdown(f"""
            **📊 Como interpretar as métricas anuais:**
            
            **Metodologia da Tese:**
            - **Total em {anos_simulacao} anos:** {formatar_br(total_evitado_tese)} tCO₂eq
            - **Média anual:** {formatar_br(media_anual_tese)} tCO₂eq/ano
            - Equivale a aproximadamente **{formatar_br(media_anual_tese / 365)} tCO₂eq/dia**
            
            **Metodologia UNFCCC:**
            - **Total em {anos_simulacao} anos:** {formatar_br(total_evitado_unfccc)} tCO₂eq
            - **Média anual:** {formatar_br(media_anual_unfccc)} tCO₂eq/ano
            - Equivale a aproximadamente **{formatar_br(media_anual_unfccc / 365)} tCO₂eq/dia**
            
            **💡 Significado prático para o Brasil (Tabela 18):**
            - As métricas anuais ajudam a planejar projetos de longo prazo
            - Permitem comparar com metas anuais de redução de emissões do Brasil
            - Facilitam o cálculo de retorno financeiro anual em Reais
            - A média anual representa o desempenho constante do projeto
            """)

        # Gráfico comparativo
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

        # Adicionar valores formatados em cima das barras
        for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Proposta da Tese'], 
                                         df_evitadas_anual['UNFCCC (2012)'])):
            ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                    formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
            ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                    formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
        ax.set_title('Comparação Anual das Emissões Evitadas: Proposta da Tese vs UNFCCC (2012)')
        
        # Ajustar o eixo x para ser igual ao do gráfico de redução acumulada
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual_revisado['Year'], fontsize=8)

        ax.legend(title='Metodologia')
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Gráfico de redução acumulada
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

        # Análise de Sensibilidade Global (Sobol) - PROPOSTA DA TESE
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Proposta da Tese")
        br_formatter_sobol = FuncFormatter(br_format)

        np.random.seed(50)  
        
        problem_tese = {
            'num_vars': 3,
            'names': ['umidade', 'T', 'DOC'],
            'bounds': [
                [0.5, 0.85],         # umidade
                [25.0, 45.0],       # temperatura
                [0.15, 0.50],       # doc
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
        ax.xaxis.set_major_formatter(br_formatter_sobol) # Adiciona formatação ao eixo x
        st.pyplot(fig)

        # Análise de Sensibilidade Global (Sobol) - CENÁRIO UNFCCC
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Cenário UNFCCC")

        np.random.seed(50)
        
        problem_unfccc = {
            'num_vars': 3,
            'names': ['umidade', 'T', 'DOC'],
            'bounds': [
                [0.5, 0.85],  # Umidade
                [25, 45],     # Temperatura
                [0.15, 0.50], # DOC
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
        ax.xaxis.set_major_formatter(br_formatter_sobol) # Adiciona formatação ao eixo x
        st.pyplot(fig)

        # Análise de Incerteza (Monte Carlo) - PROPOSTA DA TESE
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

        # Análise de Incerteza (Monte Carlo) - CENÁRIO UNFCCC
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

        # =============================================================================
        # ANÁLISE TÉCNICO-ECONÔMICA (NOVA SEÇÃO) - BRASIL (TABELA 18)
        # =============================================================================
        
        st.markdown("---")
        st.header("🏭 Análise Técnico-Econômica Integrada - Brasil (Tabela 18)")
        
        # Obter parâmetros TEA da session state
        parametros_tea = {
            'fator_capex': st.session_state.get('fator_capex', 1.0),
            'fator_opex': st.session_state.get('fator_opex', 1.0),
            'mercado_carbono': st.session_state.get('mercado_carbono', "Híbrido (Média R$ 290,82)"),
            'preco_humus': st.session_state.get('preco_humus', 10.0),  # R$ 10,00 (Tabela 18)
            'preco_minhoca': st.session_state.get('preco_minhoca', 100.0),  # R$ 100,00 (Tabela 18)
            'taxa_desconto': st.session_state.get('taxa_desconto', 0.08),
            'custo_aterro': st.session_state.get('custo_aterro', 0.30)
        }
        
        # Calcular redução anual média
        reducao_anual_tese = media_anual_tese
        reducao_anual_unfccc = media_anual_unfccc
        
        # Calcular custos - FUNÇÃO BRASILEIRA (TABELA 18)
        custos_tese = calcular_custos_capex_opex_brasil(residuos_kg_dia, anos_simulacao)
        
        # Ajustar custos com fatores da sidebar
        custos_tese['capex_total'] *= parametros_tea['fator_capex']
        custos_tese['opex_anual'] *= parametros_tea['fator_opex']
        
        # Determinar preço do carbono baseado na seleção - VALORES BRASILEIROS
        mercado_selecionado = parametros_tea['mercado_carbono']
        if mercado_selecionado == "Voluntário (R$ 37,40)":
            preco_carbono_tea = 37.40  # Mercado voluntário
        elif mercado_selecionado == "Regulado EU ETS (R$ 544,23)":
            preco_carbono_tea = 544.23  # Mercado regulado EU ETS
        elif mercado_selecionado == "Customizado":
            preco_carbono_tea = st.session_state.get('preco_carbono_custom', 290.82)
        else:  # Híbrido
            preco_carbono_tea = 290.82
        
        # Calcular receitas - FUNÇÃO BRASILEIRA (TABELA 18)
        receitas_tese = calcular_receitas_brasil(
            residuos_kg_dia, 
            reducao_anual_tese,
            preco_carbono_tea,
            mercado='regulado' if preco_carbono_tea > 500 else 'voluntario',
            preco_humus=parametros_tea['preco_humus'],
            preco_minhoca=parametros_tea['preco_minhoca'],
            custo_aterro=parametros_tea['custo_aterro']
        )
        
        # Calcular indicadores financeiros
        indicadores_tese = calcular_indicadores_financeiros(
            custos_tese['capex_total'],
            custos_tese['opex_anual'],
            receitas_tese['receita_total_anual'],
            anos_simulacao,
            parametros_tea['taxa_desconto']
        )
        
        # Análise de sensibilidade - FUNÇÃO BRASILEIRA (TABELA 18)
        sensibilidade_tese = analise_sensibilidade_tea_brasil(
            residuos_kg_dia, 
            reducao_anual_tese, 
            anos_simulacao,
            preco_humus=parametros_tea['preco_humus'],
            preco_minhoca=parametros_tea['preco_minhoca'],
            custo_aterro=parametros_tea['custo_aterro']
        )
        
        # Consolidar análise TEA
        analise_tea_completa = {
            'capex_total': custos_tese['capex_total'],
            'opex_anual': custos_tese['opex_anual'],
            'capex_detalhado': custos_tese['capex_detalhado'],
            'opex_detalhado': custos_tese['opex_detalhado'],
            'receitas': receitas_tese,
            'indicadores': indicadores_tese,
            'info_sistema': custos_tese['info_sistema']
        }
        
        # Exibir dashboard TEA BRASILEIRO (TABELA 18)
        # Nota: A função criar_dashboard_tea_brasil não está definida no código fornecido
        # Vou criar uma versão simplificada para exibir os resultados
        st.subheader("🏭 Análise Técnico-Econômica - Contexto Brasileiro (Tabela 18)")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CAPEX (Investimento)",
                f"R$ {formatar_br(analise_tea_completa['capex_total'])}",
                help="Custo total de investimento inicial"
            )
        
        with col2:
            st.metric(
                "OPEX Anual",
                f"R$ {formatar_br(analise_tea_completa['opex_anual'])}/ano",
                help="Custo operacional anual"
            )
        
        with col3:
            st.metric(
                "Receita Anual",
                f"R$ {formatar_br(analise_tea_completa['receitas']['receita_total_anual'])}/ano",
                help="Receita total anual"
            )
        
        with col4:
            vpl = analise_tea_completa['indicadores']['vpl']
            st.metric(
                "VPL (Valor Presente Líquido)",
                f"R$ {formatar_br(vpl)}",
                delta="Viável" if vpl > 0 else "Não Viável",
                delta_color="normal" if vpl > 0 else "inverse"
            )
        
        # Detalhamento das receitas
        st.subheader("💰 Detalhamento das Receitas Anuais")
        
        receitas = analise_tea_completa['receitas']
        df_receitas = pd.DataFrame({
            'Fonte de Receita': ['Húmus', 'Minhocas', 'Créditos de Carbono', 'Economia Aterro', 'Benefícios Indiretos', 'TOTAL'],
            'Valor (R$/ano)': [
                formatar_br(receitas['receita_humus']),
                formatar_br(receitas['receita_minhocas']),
                formatar_br(receitas['receita_carbono']),
                formatar_br(receitas['economia_aterro']),
                formatar_br(receitas['beneficios_indiretos']),
                formatar_br(receitas['receita_total_anual'])
            ],
            'Participação (%)': [
                f"{receitas['estrutura_receitas']['humus_perc']:.1f}",
                f"{receitas['estrutura_receitas']['minhocas_perc']:.1f}",
                f"{receitas['estrutura_receitas']['carbono_perc']:.1f}",
                f"{receitas['estrutura_receitas']['economia_aterro_perc']:.1f}",
                f"{receitas['estrutura_receitas']['beneficios_indiretos_perc']:.1f}",
                "100,0"
            ]
        })
        
        st.dataframe(df_receitas, use_container_width=True)
        
        # Gráfico de pizza das receitas
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = ['Húmus', 'Minhocas', 'Créditos Carbono', 'Economia Aterro', 'Benefícios Indiretos']
        sizes = [
            receitas['estrutura_receitas']['humus_perc'],
            receitas['estrutura_receitas']['minhocas_perc'],
            receitas['estrutura_receitas']['carbono_perc'],
            receitas['estrutura_receitas']['economia_aterro_perc'],
            receitas['estrutura_receitas']['beneficios_indiretos_perc']
        ]
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribuição das Receitas Anuais (%) - Tabela 18')
        st.pyplot(fig)
        
        # Análise de sensibilidade
        st.subheader("🎯 Análise de Sensibilidade - Cenários Brasileiros (Baseado na Tabela 18)")
        
        # Tabela comparativa de cenários
        dados_cenarios = []
        for cenario, dados in sensibilidade_tese.items():
            if dados['indicadores']['vpl'] is not None:
                roi = (dados['indicadores']['vpl'] / dados['capex']) * 100 if dados['capex'] > 0 else 0
            else:
                roi = 0
                
            dados_cenarios.append({
                'Cenário': cenario,
                'Mercado Carbono': dados['receitas_detalhadas']['mercado_selecionado'].capitalize(),
                'Preço Carbono (R$/tCO₂eq)': formatar_br(dados['receitas_detalhadas']['preco_credito_usado']),
                'CAPEX (R$)': formatar_br(dados['capex']),
                'VPL (R$)': formatar_br(dados['indicadores']['vpl']),
                'TIR (%)': f"{dados['indicadores']['tir']*100:.1f}" if dados['indicadores']['tir'] is not None else 'N/A',
                'Payback (anos)': dados['indicadores']['payback_anos'] or '>20',
                'ROI (%)': f"{roi:.1f}",
                'Viabilidade': '✅' if dados['indicadores']['vpl'] > 0 else '❌'
            })
        
        df_cenarios = pd.DataFrame(dados_cenarios)
        st.dataframe(df_cenarios, use_container_width=True)
        
        # =========================================================================
        # RESUMO EXECUTIVO TEA - BRASIL (TABELA 18)
        # =========================================================================
        
        with st.expander("📋 Resumo Executivo TEA - Brasil (Tabelas 18-19)", expanded=True):
            st.markdown(f"""
            ## 📊 Resumo Executivo - Análise Técnico-Econômica (Brasil - Tabelas 18-19)
            
            **🇧🇷 Contexto Brasileiro (Tabela 18 - Adaptação de Zziwa et al., 2021):**
            - **Escala:** {residuos_kg_dia} kg/dia ({formatar_br(residuos_kg_dia * 365 / 1000)} ton/ano)
            - **Reatores necessários:** {custos_tese['info_sistema']['num_reatores']} unidades
            - **Minhocas iniciais:** {formatar_br(custos_tese['info_sistema']['kg_minhocas'])} kg
            
            **💼 Viabilidade Financeira (Tabela 18):**
            - **VPL:** R$ {formatar_br(indicadores_tese['vpl'])} 
            - **TIR:** {f"{indicadores_tese['tir']*100:.1f}%" if indicadores_tese['tir'] is not None else 'N/A'}
            - **Payback:** {indicadores_tese['payback_anos'] or '> período'} anos
            - **Custo por tCO₂eq evitada:** R$ {formatar_br(indicadores_tese['custo_tonelada_evitada'])}
            
            **💰 Estrutura de Custos e Receitas (R$) - Tabela 18:**
            - **Investimento (CAPEX):** R$ {formatar_br(custos_tese['capex_total'])}
            - **Custo Anual (OPEX):** R$ {formatar_br(custos_tese['opex_anual'])}/ano
            - **Receita Total Anual:** R$ {formatar_br(receitas_tese['receita_total_anual'])}/ano
              - **Húmus:** R$ {formatar_br(receitas_tese['receita_humus'])}/ano ({receitas_tese['producao_humus_kg_ano']/1000:.1f} ton × R$ {receitas_tese['parametros_entrada']['preco_humus']}/kg)
              - **Minhocas:** R$ {formatar_br(receitas_tese['receita_minhocas'])}/ano ({receitas_tese['producao_minhocas_kg_ano']:.0f} kg × R$ {receitas_tese['parametros_entrada']['preco_minhoca']}/kg)
              - **Créditos de Carbono:** R$ {formatar_br(receitas_tese['receita_carbono'])}/ano
              - **Economia com Aterro:** R$ {formatar_br(receitas_tese['economia_aterro'])}/ano
              - **Benefícios Indiretos:** R$ {formatar_br(receitas_tese['beneficios_indiretos'])}/ano
            
            **🌍 Impacto Econômico-Ambiental (Tabela 19):**
            - **Custo de Abatimento:** R$ {formatar_br(indicadores_tese['custo_tonelada_evitada'])}/tCO₂eq
            - **Preço de Mercado:** R$ {formatar_br(preco_carbono_tea)}/tCO₂eq
            - **Margem por Crédito:** R$ {formatar_br(preco_carbono_tea - indicadores_tese['custo_tonelada_evitada'])}
            - **Produção de Húmus:** {formatar_br(receitas_tese['producao_humus_kg_ano']/1000)} ton/ano
            - **Produção de Minhocas:** {formatar_br(receitas_tese['producao_minhocas_kg_ano'])} kg/ano
            
            **🎯 Cenários de Mercado para Brasil (Tabela 19):**
            - **Voluntário (R$ 37,40):** VPL = R$ {formatar_br(sensibilidade_tese['Pessimista (Voluntário)']['indicadores']['vpl'])}
            - **Híbrido (R$ 290,82):** VPL = R$ {formatar_br(sensibilidade_tese['Realista (Híbrido)']['indicadores']['vpl'])}
            - **Regulado EU ETS (R$ 544,23):** VPL = R$ {formatar_br(sensibilidade_tese['Otimista (Regulado EU ETS)']['indicadores']['vpl'])}
            
            **⚖️ Conclusão TEA para Brasil (Tabela 18):**
            {"✅ **PROJETO VIÁVEL** - VPL positivo e TIR acima do custo de capital" 
             if indicadores_tese['vpl'] > 0 else 
             "⚠️ **PROJETO NÃO VIÁVEL** - Necessita de ajustes ou incentivos"}
            """)
        
        # =============================================================================
        # RODAPÉ ATUALIZADO COM REFERÊNCIAS DAS TABELAS 18-19
        # =============================================================================

        # Rodapé
        st.markdown("---")
        st.markdown("""

        **📚 Referências por Cenário - Brasil (Tabelas 18-19):**

        **Cenário de Baseline (Aterro Sanitário) - Brasil:**
        - Metano: IPCC (2006), UNFCCC (2016) e Wang et al. (2023) adaptado
        - Óxido Nitroso: Wang et al. (2017) adaptado
        - Metano e Óxido Nitroso no pré-descarte: Feng et al. (2020) adaptado
        - Custos de disposição: ABRELPE (2024) - R$ 300/ton

        **Proposta da Tese (Compostagem em reatores com minhocas) - Brasil (Tabela 18):**
        - Metano e Óxido Nitroso: Yang et al. (2017) adaptado
        - Custos de investimento: Tabela 18 - Zziwa et al. (2021) adaptado para Brasil
        - Reatores: {residuos_kg_dia} unidades × R$ 1.000 = R$ {formatar_br(residuos_kg_dia*1000)}
        - Minhocas: {residuos_kg_dia*3} kg × R$ 100 = R$ {formatar_br(residuos_kg_dia*300)}
        - Receitas: Húmus R$ 10,00/kg, Minhocas R$ 100,00/kg

        **Cenário UNFCCC (Compostagem sem minhocas a céu aberto) - Brasil:**
        - Protocolo AMS-III.F: UNFCCC (2016)
        - Fatores de emissões: Yang et al. (2017)

        **🌍 Mercados de Carbono - Contexto Brasileiro (Tabela 19):**
        - **Mercado Voluntário:** State of Voluntary Carbon Markets 2024 (USD 7.45/tCO₂eq ≈ R$ 37,40)
        - **Mercado Regulado:** EU ETS Futures Dec/2025 (€72.29/tCO₂eq ≈ R$ 461,20)
        - **Câmbio T19 (maio/2025):** EUR/BRL: 6,38; USD/BRL: 5,65
        - **Adaptação econômica:** Valores convertidos para Real Brasileiro (R$)

        **🇧🇷 Contextualização para o Brasil (Tabela 18):**
        - Escala: {residuos_kg_dia} kg/dia = {residuos_kg_dia*365/1000:.1f} ton/ano
        - Reatores necessários: {residuos_kg_dia} unidades (1 reator/kg/dia)
        - Minhocas iniciais: {residuos_kg_dia*3} kg (3 kg/kg/dia)
        - Produção anual de húmus: {residuos_kg_dia*365/1000*0.4:.1f} ton (40% dos resíduos)
        - Produção anual de minhocas: {7.45*residuos_kg_dia:.1f} kg (7,45 kg/kg/dia)
        - Receita total sem carbono (Tabela 18): R$ {formatar_br(residuos_kg_dia*365/1000*0.4*1000*10 + 7.45*residuos_kg_dia*100 + residuos_kg_dia*365/1000*300)}
        """)

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação Completa' para ver os resultados baseados na Tabela 18 da tese.")
