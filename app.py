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

st.set_page_config(page_title="Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class GHGEmissionCalculator:
    def __init__(self):
        self.TOC = 0.436
        self.TN = 0.0142
        self.f_CH4_vermi = 0.0013
        self.f_N2O_vermi = 0.0092
        self.f_CH4_thermo = 0.0060
        self.f_N2O_thermo = 0.0196
        self.COMPOSTING_DAYS = 50
        self.GWP_CH4_20 = 79.7
        self.GWP_N2O_20 = 273
        self.MCF = 1.0
        self.F = 0.5
        self.OX = 0.1
        self.Ri = 0.0
        self._load_emission_profiles()
        self._setup_pre_disposal_emissions()
    
    def _load_emission_profiles(self):
        self.profile_ch4_vermi = np.array([
            0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
            0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
            0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_ch4_vermi /= self.profile_ch4_vermi.sum()
        
        self.profile_n2o_vermi = np.array([
            0.15, 0.10, 0.20, 0.05, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06,
            0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
            0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_vermi /= self.profile_n2o_vermi.sum()
        
        self.profile_ch4_thermo = self.profile_ch4_vermi.copy()
        
        self.profile_n2o_thermo = np.array([
            0.10, 0.08, 0.15, 0.05, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12,
            0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05,
            0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_thermo /= self.profile_n2o_thermo.sum()
        
        self.profile_n2o_landfill = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
    
    def _setup_pre_disposal_emissions(self):
        CH4_pre_ugC_per_kg_h = 2.78
        self.CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000
        
        N2O_pre_mgN_per_kg = 20.26
        N2O_pre_mgN_per_kg_day = N2O_pre_mgN_per_kg / 3
        self.N2O_pre_kg_per_kg_day = N2O_pre_mgN_per_kg_day * (44/28) / 1_000_000
        
        self.profile_n2o_pre = {1: 0.8623, 2: 0.10, 3: 0.0377}
    
    def calculate_landfill_emissions(self, waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years=20):
        days = years * 365
        docf = 0.0147 * temperature_C + 0.28
        ch4_potential_per_kg = (doc_fraction * docf * self.MCF * self.F * (16/12) * (1 - self.Ri) * (1 - self.OX))
        ch4_potential_daily = waste_kg_day * ch4_potential_per_kg
        
        t = np.arange(1, days + 1, dtype=float)
        kernel_ch4 = np.exp(-k_year * (t - 1) / 365.0) - np.exp(-k_year * t / 365.0)
        daily_inputs = np.ones(days, dtype=float)
        ch4_emissions = fftconvolve(daily_inputs, kernel_ch4, mode='full')[:days]
        ch4_emissions *= ch4_potential_daily
        
        exposed_mass = 100
        exposed_hours = 8
        opening_factor = (exposed_mass / waste_kg_day) * (exposed_hours / 24)
        opening_factor = np.clip(opening_factor, 0.0, 1.0)
        
        E_open = 1.91
        E_closed = 2.15
        E_avg = opening_factor * E_open + (1 - opening_factor) * E_closed
        
        moisture_factor = (1 - moisture_fraction) / (1 - 0.55)
        E_avg_adjusted = E_avg * moisture_factor
        
        daily_n2o_kg = (E_avg_adjusted * (44/28) / 1_000_000) * waste_kg_day
        
        kernel_n2o = np.array([self.profile_n2o_landfill.get(d, 0) for d in range(1, 6)], dtype=float)
        n2o_emissions = fftconvolve(np.full(days, daily_n2o_kg), kernel_n2o, mode='full')[:days]
        
        ch4_pre, n2o_pre = self._calculate_pre_disposal(waste_kg_day, days)
        
        return ch4_emissions + ch4_pre, n2o_emissions + n2o_pre
    
    def _calculate_pre_disposal(self, waste_kg_day, days):
        ch4_emissions = np.full(days, waste_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o_emissions = np.zeros(days)
        
        for entry_day in range(days):
            for days_after, fraction in self.profile_n2o_pre.items():
                emission_day = entry_day + days_after - 1
                if emission_day < days:
                    n2o_emissions[emission_day] += (waste_kg_day * self.N2O_pre_kg_per_kg_day * fraction)
        
        return ch4_emissions, n2o_emissions
    
    def calculate_vermicomposting_emissions(self, waste_kg_day, moisture_fraction, years=20, f_ch4=None, f_n2o=None):
        if f_ch4 is None:
            f_ch4 = self.f_CH4_vermi
        if f_n2o is None:
            f_n2o = self.f_N2O_vermi
        days = years * 365
        dry_fraction = 1 - moisture_fraction
        
        ch4_per_batch = (waste_kg_day * self.TOC * f_ch4 * (16/12) * dry_fraction)
        n2o_per_batch = (waste_kg_day * self.TN * f_n2o * (44/28) * dry_fraction)
        
        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)
        
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_per_batch * self.profile_ch4_vermi[compost_day]
                    n2o_emissions[emission_day] += n2o_per_batch * self.profile_n2o_vermi[compost_day]
        
        return ch4_emissions, n2o_emissions
    
    def calculate_thermophilic_emissions(self, waste_kg_day, moisture_fraction, years=20):
        days = years * 365
        dry_fraction = 1 - moisture_fraction
        
        ch4_per_batch = (waste_kg_day * self.TOC * self.f_CH4_thermo * (16/12) * dry_fraction)
        n2o_per_batch = (waste_kg_day * self.TN * self.f_N2O_thermo * (44/28) * dry_fraction)
        
        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)
        
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_per_batch * self.profile_ch4_thermo[compost_day]
                    n2o_emissions[emission_day] += n2o_per_batch * self.profile_n2o_thermo[compost_day]
        
        return ch4_emissions, n2o_emissions
    
    def calculate_avoided_emissions(self, waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years=20,
                                    gwp_ch4=None, gwp_n2o=None, f_ch4_vermi=None, f_n2o_vermi=None):
        if gwp_ch4 is None:
            gwp_ch4 = self.GWP_CH4_20
        if gwp_n2o is None:
            gwp_n2o = self.GWP_N2O_20
        if f_ch4_vermi is None:
            f_ch4_vermi = self.f_CH4_vermi
        if f_n2o_vermi is None:
            f_n2o_vermi = self.f_N2O_vermi

        ch4_landfill, n2o_landfill = self.calculate_landfill_emissions(
            waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years
        )
        
        ch4_vermi, n2o_vermi = self.calculate_vermicomposting_emissions(
            waste_kg_day, moisture_fraction, years, f_ch4=f_ch4_vermi, f_n2o=f_n2o_vermi
        )
        
        ch4_thermo, n2o_thermo = self.calculate_thermophilic_emissions(
            waste_kg_day, moisture_fraction, years
        )
        
        baseline_co2eq = (ch4_landfill * gwp_ch4 + n2o_landfill * gwp_n2o) / 1000
        vermi_co2eq = (ch4_vermi * gwp_ch4 + n2o_vermi * gwp_n2o) / 1000
        thermo_co2eq = (ch4_thermo * gwp_ch4 + n2o_thermo * gwp_n2o) / 1000
        
        avoided_vermi = baseline_co2eq.sum() - vermi_co2eq.sum()
        avoided_thermo = baseline_co2eq.sum() - thermo_co2eq.sum()
        
        results = {
            'baseline': {
                'ch4_kg': ch4_landfill.sum(),
                'n2o_kg': n2o_landfill.sum(),
                'co2eq_t': baseline_co2eq.sum()
            },
            'vermicomposting': {
                'ch4_kg': ch4_vermi.sum(),
                'n2o_kg': n2o_vermi.sum(),
                'co2eq_t': vermi_co2eq.sum(),
                'avoided_co2eq_t': avoided_vermi
            },
            'thermophilic': {
                'ch4_kg': ch4_thermo.sum(),
                'n2o_kg': n2o_thermo.sum(),
                'co2eq_t': thermo_co2eq.sum(),
                'avoided_co2eq_t': avoided_thermo
            },
            'comparison': {
                'difference_tco2eq': avoided_vermi - avoided_thermo,
                'superiority_percent': ((avoided_vermi / avoided_thermo) - 1) * 100 if avoided_thermo != 0 else 0
            },
            'annual_averages': {
                'baseline_tco2eq_year': baseline_co2eq.sum() / years,
                'vermi_avoided_year': avoided_vermi / years,
                'thermo_avoided_year': avoided_thermo / years
            }
        }
        
        return results

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
    
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

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
    
    return 5.50, "R$", False, "Referência"

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
        value=f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {formatar_br(st.session_state.taxa_cambio)}
        - **Carbono em Reais:** R$ {formatar_br(preco_carbono_reais)}/tCO₂eq
        
        **🌍 Mercado de Referência:**
        - European Union Allowances (EUA)
        - European Emissions Trading System (EU ETS)
        - Contratos futuros de carbono
        - Preços em tempo real
        
        **🔄 Atualização:**
        - As cotações são carregadas automaticamente ao abrir o aplicativo
        - Clique em **"Atualizar Cotações"** para obter valores mais recentes
        - Em caso de falha na conexão, são utilizados valores de referência atualizados
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
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
    if 'k_ano' not in st.session_state:
        st.session_state.k_ano = 0.06

inicializar_session_state()

def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    
    numero = round(numero, 2)
    
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_br_dec(numero, decimais=2):
    if pd.isna(numero):
        return "N/A"
    
    numero = round(numero, decimais)
    
    return f"{numero:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

st.title("Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global")
st.markdown("Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para dois contextos de gestão de resíduos")

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", min_value=10, max_value=1000, value=100, step=10)
    
    st.subheader("📊 Parâmetros Determinísticos")
    st.info("Estes valores são usados no cálculo determinístico e como valores nominais nas análises de sensibilidade/incerteza.")
    
    opcao_k = st.selectbox(
        "Taxa de decaimento do aterro (k) [ano⁻¹]",
        options=[
            "k = 0.06 (decaimento lento - valor padrão)",
            "k = 0.40 (decaimento rápido)"
        ],
        index=0
    )
    
    if "0.40" in opcao_k:
        k_ano = 0.40
    else:
        k_ano = 0.06
    
    st.session_state.k_ano = k_ano
    st.write(f"**Valor selecionado:** {formatar_br(k_ano)} ano⁻¹")
    
    T = st.slider("Temperatura média (°C)", min_value=20, max_value=40, value=25, step=1)
    st.write(f"**Valor selecionado:** {formatar_br(T)} °C")
    
    DOC = st.slider("DOC (fração)", min_value=0.10, max_value=0.25, value=0.15, step=0.01)
    st.write(f"**Valor selecionado:** {formatar_br(DOC)}")
    
    umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0
    st.write(f"**Valor fixo:** {formatar_br(umidade_valor)}%")
    
    with st.expander("ℹ️ Parâmetros variados nas análises de sensibilidade e incerteza"):
        st.markdown("""
        **Análise de Sobol e Monte Carlo** variam os seguintes parâmetros:
        - **Temperatura (T):** 20–30 °C (uniforme)
        - **Umidade (U):** 55–85% (uniforme)
        - **fCH₄ vermicompostagem:** 0,000107–0,0013 (uniforme)
        - **fN₂O vermicompostagem:** 0,000739–0,0092 (uniforme)
        - **GWP_CH₄:** 7,2–79,7 (probabilístico por cenários)
        - **GWP_N₂O:** 130–273 (probabilístico por cenários)
        
        **Cenários de GWP (probabilidades):**
        - Otimista (GWP‑20): CH₄=79,7; N₂O=273 (30%)
        - Realista (GWP‑100): CH₄=27,0; N₂O=273 (50%)
        - Pessimista (GWP‑500): CH₄=7,2; N₂O=130 (20%)
        """)
    
    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)
    
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# ------------------------------------------------------------------------------
# FUNÇÕES PARA ANÁLISE DE SENSIBILIDADE E INCERTEZA (conforme script do Colab)
# ------------------------------------------------------------------------------
def executar_simulacao_sobol_vermi(params):
    T, U, fCH4, fN2O, GWP_CH4, GWP_N2O = params
    # Usa os valores determinísticos selecionados para k e DOC (fixos)
    calc = GHGEmissionCalculator()
    res = calc.calculate_avoided_emissions(
        waste_kg_day=residuos_kg_dia,
        k_year=k_ano,
        temperature_C=T,
        doc_fraction=DOC,
        moisture_fraction=U/100.0,
        years=anos_simulacao,
        gwp_ch4=GWP_CH4,
        gwp_n2o=GWP_N2O,
        f_ch4_vermi=fCH4,
        f_n2o_vermi=fN2O
    )
    return res['vermicomposting']['avoided_co2eq_t']

def executar_simulacao_sobol_thermo(params):
    T, U, fCH4, fN2O, GWP_CH4, GWP_N2O = params
    calc = GHGEmissionCalculator()
    # Para termofílica, as frações fCH4 e fN2O são fixas (da classe), ignoramos os valores amostrados
    res = calc.calculate_avoided_emissions(
        waste_kg_day=residuos_kg_dia,
        k_year=k_ano,
        temperature_C=T,
        doc_fraction=DOC,
        moisture_fraction=U/100.0,
        years=anos_simulacao,
        gwp_ch4=GWP_CH4,
        gwp_n2o=GWP_N2O
        # não passamos f_ch4_vermi, f_n2o_vermi, então usa os padrões da classe
    )
    return res['thermophilic']['avoided_co2eq_t']

def gerar_parametros_mc(n):
    np.random.seed(50)
    T_vals = np.random.uniform(20.0, 30.0, n)
    U_vals = np.random.uniform(55.0, 85.0, n)
    fCH4_vals = np.random.uniform(0.000107, 0.0013, n)
    fN2O_vals = np.random.uniform(0.000739, 0.0092, n)
    
    # Cenários GWP (probabilístico)
    cenarios = np.random.choice(['otimista', 'real', 'pessimista'], size=n, p=[0.3, 0.5, 0.2])
    gwp_ch4_map = {'otimista': 79.7, 'real': 27.0, 'pessimista': 7.2}
    gwp_n2o_map = {'otimista': 273, 'real': 273, 'pessimista': 130}
    GWP_CH4_vals = np.array([gwp_ch4_map[c] for c in cenarios])
    GWP_N2O_vals = np.array([gwp_n2o_map[c] for c in cenarios])
    
    return T_vals, U_vals, fCH4_vals, fN2O_vals, GWP_CH4_vals, GWP_N2O_vals, cenarios

# ------------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ------------------------------------------------------------------------------
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        calculator = GHGEmissionCalculator()
        k_ano = st.session_state.k_ano
        
        # Cálculo determinístico (com os valores nominais)
        results = calculator.calculate_avoided_emissions(
            waste_kg_day=residuos_kg_dia,
            k_year=k_ano,
            temperature_C=T,
            doc_fraction=DOC,
            moisture_fraction=umidade,
            years=anos_simulacao
        )
        
        dias = anos_simulacao * 365
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')
        
        ch4_aterro_dia, n2o_aterro_dia = calculator.calculate_landfill_emissions(
            residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao
        )
        
        ch4_vermi_dia, n2o_vermi_dia = calculator.calculate_vermicomposting_emissions(
            residuos_kg_dia, umidade, anos_simulacao
        )
        
        df = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_aterro_dia,
            'N2O_Aterro_kg_dia': n2o_aterro_dia,
            'CH4_Vermi_kg_dia': ch4_vermi_dia,
            'N2O_Vermi_kg_dia': n2o_vermi_dia,
        })
        
        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi']:
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * (calculator.GWP_CH4_20 if 'CH4' in gas else calculator.GWP_N2O_20) / 1000
        
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
        
        ch4_compost_dia, n2o_compost_dia = calculator.calculate_thermophilic_emissions(
            residuos_kg_dia, umidade, anos_simulacao
        )
        
        ch4_compost_unfccc_tco2eq = ch4_compost_dia * calculator.GWP_CH4_20 / 1000
        n2o_compost_unfccc_tco2eq = n2o_compost_dia * calculator.GWP_N2O_20 / 1000
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
        
        st.info(f"""
        **Parâmetros utilizados na simulação:**
        - **Taxa de decaimento (k):** {formatar_br(k_ano)} ano⁻¹
        - **Temperatura (T):** {formatar_br(T)} °C
        - **DOC:** {formatar_br(DOC)}
        - **Umidade:** {formatar_br(umidade_valor)}%
        - **Resíduos/dia:** {formatar_br(residuos_kg_dia)} kg
        - **Total de resíduos:** {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} toneladas
        """)
        
        total_evitado_tese = results['vermicomposting']['avoided_co2eq_t']
        total_evitado_unfccc = results['thermophilic']['avoided_co2eq_t']
        
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
                f"{moeda} {formatar_br(preco_carbono)}/tCO₂eq",
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
        
        with st.expander("💡 Como funciona a comercialização no mercado de carbono?"):
            st.markdown(f"""
            **📊 Informações de Mercado:**
            - **Preço em Euro:** {moeda} {formatar_br(preco_carbono)}/tCO₂eq
            - **Preço em Real:** R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
            - **Taxa de câmbio:** 1 Euro = R$ {formatar_br(taxa_cambio)}
            - **Fonte:** {fonte_cotacao}
            
            **💶 Comprar créditos (compensação):**
            - Custo em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
            - Custo em Real: **R$ {formatar_br(valor_tese_brl)}**
            
            **💵 Vender créditos (comercialização):**  
            - Receita em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
            - Receita em Real: **R$ {formatar_br(valor_tese_brl)}**
            
            **🌍 Mercado de Referência:**
            - European Union Allowances (EUA)
            - European Emissions Trading System (EU ETS)
            - Contratos futuros de carbono
            - Preços em tempo real do mercado regulado
            """)
        
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
        ax.set_title('Redução de Emissões em {} Anos (k = {} ano⁻¹)'.format(anos_simulacao, formatar_br(k_ano)))
        ax.set_xlabel('Ano')
        ax.set_ylabel('tCO₂eq Acumulado')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(br_formatter)

        st.pyplot(fig)

        # ------------------------------------------------------------------
        # ANÁLISE DE SENSIBILIDADE SOBOL (6 parâmetros)
        # ------------------------------------------------------------------
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Proposta da Tese")
        st.info("**Parâmetros variados:** T, U, fCH₄, fN₂O, GWP_CH₄, GWP_N₂O (distribuições uniformes nos intervalos descritos).")
        br_formatter_sobol = FuncFormatter(br_format)

        np.random.seed(50)
        
        problem_sobol = {
            'num_vars': 6,
            'names': ['T', 'U', 'fCH4', 'fN2O', 'GWP_CH4', 'GWP_N2O'],
            'bounds': [
                [20.0, 30.0],
                [55.0, 85.0],
                [0.000107, 0.0013],
                [0.000739, 0.0092],
                [7.2, 79.7],
                [130.0, 273.0]
            ]
        }

        param_values_sobol = sample(problem_sobol, n_samples, seed=50)
        # Executa paralelamente para vermicompostagem
        results_sobol_vermi = Parallel(n_jobs=-1)(delayed(executar_simulacao_sobol_vermi)(params) for params in param_values_sobol)
        Si_vermi = analyze(problem_sobol, np.array(results_sobol_vermi), print_to_console=False)
        
        # Para termofílica (usando os mesmos parâmetros, mas fCH4 e fN2O não influenciam)
        results_sobol_thermo = Parallel(n_jobs=-1)(delayed(executar_simulacao_sobol_thermo)(params) for params in param_values_sobol)
        Si_thermo = analyze(problem_sobol, np.array(results_sobol_thermo), print_to_console=False)
        
        # DataFrame para vermicompostagem
        sensibilidade_df_vermi = pd.DataFrame({
            'Parâmetro': problem_sobol['names'],
            'S1': Si_vermi['S1'],
            'ST': Si_vermi['ST']
        }).sort_values('ST', ascending=False)
        
        nomes_amigaveis = {
            'T': 'Temperatura (°C)',
            'U': 'Umidade (%)',
            'fCH4': 'fCH₄ Vermi',
            'fN2O': 'fN₂O Vermi',
            'GWP_CH4': 'GWP CH₄',
            'GWP_N2O': 'GWP N₂O'
        }
        sensibilidade_df_vermi['Parâmetro'] = sensibilidade_df_vermi['Parâmetro'].map(nomes_amigaveis)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_vermi, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global - Proposta da Tese (Índice Total ST)')
        ax.set_xlabel('Índice ST (Sobol Total)')
        ax.set_ylabel('Parâmetro')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        
        for i, (st_value) in enumerate(sensibilidade_df_vermi['ST']):
            ax.text(st_value, i, f' {formatar_br(st_value)}', va='center', fontweight='bold')
        
        st.pyplot(fig)
        
        st.subheader("📊 Valores de Sensibilidade - Proposta da Tese")
        st.dataframe(sensibilidade_df_vermi.style.format({
            'S1': '{:.4f}',
            'ST': '{:.4f}'
        }))

        # Gráfico para termofílica
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Cenário UNFCCC")
        
        sensibilidade_df_thermo = pd.DataFrame({
            'Parâmetro': problem_sobol['names'],
            'S1': Si_thermo['S1'],
            'ST': Si_thermo['ST']
        }).sort_values('ST', ascending=False)
        sensibilidade_df_thermo['Parâmetro'] = sensibilidade_df_thermo['Parâmetro'].map(nomes_amigaveis)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_thermo, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global - Cenário UNFCCC (Índice Total ST)')
        ax.set_xlabel('Índice ST (Sobol Total)')
        ax.set_ylabel('Parâmetro')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        
        for i, (st_value) in enumerate(sensibilidade_df_thermo['ST']):
            ax.text(st_value, i, f' {formatar_br(st_value)}', va='center', fontweight='bold')
        
        st.pyplot(fig)
        
        st.subheader("📊 Valores de Sensibilidade - Cenário UNFCCC")
        st.dataframe(sensibilidade_df_thermo.style.format({
            'S1': '{:.4f}',
            'ST': '{:.4f}'
        }))

        # ------------------------------------------------------------------
        # ANÁLISE DE INCERTEZA MONTE CARLO (6 parâmetros)
        # ------------------------------------------------------------------
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Proposta da Tese")
        
        T_vals, U_vals, fCH4_vals, fN2O_vals, GWP_CH4_vals, GWP_N2O_vals, cenarios = gerar_parametros_mc(n_simulations)
        
        results_mc_tese = []
        for i in range(n_simulations):
            calc_mc = GHGEmissionCalculator()
            res_mc = calc_mc.calculate_avoided_emissions(
                waste_kg_day=residuos_kg_dia,
                k_year=k_ano,
                temperature_C=T_vals[i],
                doc_fraction=DOC,
                moisture_fraction=U_vals[i]/100.0,
                years=anos_simulacao,
                gwp_ch4=GWP_CH4_vals[i],
                gwp_n2o=GWP_N2O_vals[i],
                f_ch4_vermi=fCH4_vals[i],
                f_n2o_vermi=fN2O_vals[i]
            )
            results_mc_tese.append(res_mc['vermicomposting']['avoided_co2eq_t'])

        results_array_tese = np.array(results_mc_tese)
        media_tese = np.mean(results_array_tese)
        p5_tese = np.percentile(results_array_tese, 5)
        p95_tese = np.percentile(results_array_tese, 95)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_tese, kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(media_tese, color='red', linestyle='--', label=f'Média: {formatar_br(media_tese)} tCO₂eq')
        ax.axvline(p5_tese, color='green', linestyle=':', label='Percentil 5')
        ax.axvline(p95_tese, color='green', linestyle=':', label='Percentil 95')
        ax.set_title('Distribuição das Emissões Evitadas (Monte Carlo) - Proposta da Tese')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Cenário UNFCCC")
        
        results_mc_thermo = []
        for i in range(n_simulations):
            calc_mc = GHGEmissionCalculator()
            res_mc = calc_mc.calculate_avoided_emissions(
                waste_kg_day=residuos_kg_dia,
                k_year=k_ano,
                temperature_C=T_vals[i],
                doc_fraction=DOC,
                moisture_fraction=U_vals[i]/100.0,
                years=anos_simulacao,
                gwp_ch4=GWP_CH4_vals[i],
                gwp_n2o=GWP_N2O_vals[i]
                # fCH4 e fN2O não são passados; usa os padrões da classe
            )
            results_mc_thermo.append(res_mc['thermophilic']['avoided_co2eq_t'])

        results_array_thermo = np.array(results_mc_thermo)
        media_thermo = np.mean(results_array_thermo)
        p5_thermo = np.percentile(results_array_thermo, 5)
        p95_thermo = np.percentile(results_array_thermo, 95)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_thermo, kde=True, bins=30, color='coral', ax=ax)
        ax.axvline(media_thermo, color='red', linestyle='--', label=f'Média: {formatar_br(media_thermo)} tCO₂eq')
        ax.axvline(p5_thermo, color='green', linestyle=':', label='Percentil 5')
        ax.axvline(p95_thermo, color='green', linestyle=':', label='Percentil 95')
        ax.set_title('Distribuição das Emissões Evitadas (Monte Carlo) - Cenário UNFCCC')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        st.subheader("📊 Análise Estatística de Comparação")
        
        diferencas = results_array_tese - results_array_thermo
        _, p_valor_normalidade_diff = stats.normaltest(diferencas)
        st.write(f"Teste de normalidade das diferenças (p-value): **{formatar_br_dec(p_valor_normalidade_diff, 5)}**")

        ttest_pareado, p_ttest_pareado = stats.ttest_rel(results_array_tese, results_array_thermo)
        st.write(f"Teste T pareado: Estatística t = **{formatar_br_dec(ttest_pareado, 5)}**, P-valor = **{formatar_br_dec(p_ttest_pareado, 5)}**")

        wilcoxon_stat, p_wilcoxon = stats.wilcoxon(results_array_tese, results_array_thermo)
        st.write(f"Teste de Wilcoxon (pareado): Estatística = **{formatar_br_dec(wilcoxon_stat, 5)}**, P-valor = **{formatar_br_dec(p_wilcoxon, 5)}**")

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
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

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

**⚠️ Nota de Reprodutibilidade:**
- Todas as análises usam seed fixo (50) para garantir resultados idênticos ao script testado no Google Colab
- Ajustados os ranges de parâmetros para DOC (0.10-0.25) e Temperatura (20-40°C)
- Métodos de cálculo idênticos aos do script testado no Google Colab
""")
