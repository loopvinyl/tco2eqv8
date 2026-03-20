# -*- coding: utf-8 -*-
"""
Nutriwash Circular System BR - Aplicativo Streamlit
Calculadora de Emissões de GEE para Tecnologias de Gestão de Resíduos
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import fftconvolve
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
from joblib import Parallel, delayed
import warnings
from datetime import datetime
from matplotlib.ticker import FuncFormatter

# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================
warnings.filterwarnings('ignore')
np.random.seed(50)  # reprodutibilidade
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

# Formatadores brasileiros (ponto de milhar e vírgula decimal)
def br_format_inteiro(x, pos):
    return f'{x:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

def br_format_decimal(x, pos):
    return f'{x:.4f}'.replace('.', ',')

# =============================================================================
# CLASSE PRINCIPAL DE CÁLCULO DE EMISSÕES (MESMA DO SCRIPT ORIGINAL)
# =============================================================================
class GHGEmissionCalculator:
    """Calculadora principal de emissões de GEE para gestão de resíduos"""
    
    def __init__(self):
        # Caracterização do resíduo (Yang et al., 2017)
        self.TOC = 0.436  # kg C / kg resíduo úmido
        self.TN = 0.0142  # kg N / kg resíduo úmido
        
        # Frações de emissão na compostagem (Yang et al., 2017)
        self.f_CH4_vermi = 0.0013   # 0,13% do TOC para vermicompostagem
        self.f_N2O_vermi = 0.0092   # 0,92% do TN para vermicompostagem
        self.f_CH4_thermo = 0.0060  # 0,60% do TOC para termofílica
        self.f_N2O_thermo = 0.0196  # 1,96% do TN para termofílica
        
        # Período de compostagem (dias)
        self.COMPOSTING_DAYS = 50
        
        # Potenciais de Aquecimento Global (padrão: cenário realista GWP-100)
        self.GWP_CH4_20 = 27.0   # GWP do CH4 (realista)
        self.GWP_N2O_20 = 273    # GWP do N2O (realista)
        
        # Parâmetros do aterro (IPCC 2006)
        self.MCF = 1.0  # Fator de correção de metano (aterro gerenciado)
        self.F = 0.5    # Fração de CH4 no gás do aterro
        self.OX = 0.1   # Fator de oxidação
        self.Ri = 0.0   # Sem recuperação
        
        # Parâmetros fixos (valores nominais)
        self.residuos_kg_dia = 100
        self.umidade = 0.85
        self.massa_exposta_kg = 100
        self.h_exposta = 8
        self.T = 25
        self.DOC = 0.15
        self.k_ano = 0.06
        
        # Carregar perfis de emissão
        self._load_emission_profiles()
        
        # Emissões de pré-descarte (Feng et al., 2020)
        self._setup_pre_disposal_emissions()
    
    def _load_emission_profiles(self):
        """Carrega e normaliza os perfis de emissão do Apêndice A"""
        # Perfil de CH4 para vermicompostagem (Apêndice A.4.1)
        self.profile_ch4_vermi = np.array([
            0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
            0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
            0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_ch4_vermi /= self.profile_ch4_vermi.sum()
        
        # Perfil de N2O para vermicompostagem (Apêndice A.4.2)
        self.profile_n2o_vermi = np.array([
            0.15, 0.10, 0.20, 0.05, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06,
            0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
            0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_vermi /= self.profile_n2o_vermi.sum()
        
        # Perfil de CH4 para compostagem termofílica (Apêndice A.4.3)
        self.profile_ch4_thermo = np.array([
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
        self.profile_ch4_thermo /= self.profile_ch4_thermo.sum()
        
        # Perfil de N2O para compostagem termofílica (Apêndice A.4.4)
        self.profile_n2o_thermo = np.array([
            0.10, 0.08, 0.15, 0.05, 0.03,
            0.04, 0.05, 0.07, 0.10, 0.12,
            0.15, 0.18, 0.20, 0.18, 0.15,
            0.12, 0.10, 0.08, 0.06, 0.05,
            0.04, 0.03, 0.02, 0.02, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002,
            0.001, 0.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_thermo /= self.profile_n2o_thermo.sum()
        
        # Perfil de N2O para aterro (Apêndice A.5)
        self.profile_n2o_landfill = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
        
        # Perfil de N2O para pré-descarte (Feng et al., 2020)
        self.profile_n2o_pre = {1: 0.8623, 2: 0.10, 3: 0.0377}
    
    def _setup_pre_disposal_emissions(self):
        """Configura fatores de emissão de pré-descarte (Feng et al., 2020)"""
        # CH4 pré-descarte
        CH4_pre_ugC_per_kg_h = 2.78
        self.CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000
        
        # N2O pré-descarte
        N2O_pre_mgN_per_kg = 20.26
        N2O_pre_mgN_per_kg_day = N2O_pre_mgN_per_kg / 3
        self.N2O_pre_kg_per_kg_day = N2O_pre_mgN_per_kg_day * (44/28) / 1_000_000
    
    def calculate_landfill_emissions(self, waste_kg_day, k_year, temperature_C, 
                                    doc_fraction, moisture_fraction, years=20):
        """Calcula as emissões do aterro usando o método FOD do IPCC."""
        days = years * 365
        
        # Cálculo do DOCf (fração que realmente se decompõe)
        docf = 0.0147 * temperature_C + 0.28
        
        # Potencial de CH4 por kg de resíduo
        ch4_potential_per_kg = (doc_fraction * docf * self.MCF * self.F * (16/12) * (1 - self.Ri) * (1 - self.OX))
        
        ch4_potential_daily = waste_kg_day * ch4_potential_per_kg
        
        # Distribuição de decaimento de primeira ordem
        t = np.arange(1, days + 1, dtype=float)
        kernel_ch4 = np.exp(-k_year * (t - 1) / 365.0) - np.exp(-k_year * t / 365.0)
        daily_inputs = np.ones(days, dtype=float)
        ch4_emissions = fftconvolve(daily_inputs, kernel_ch4, mode='full')[:days]
        ch4_emissions *= ch4_potential_daily
        
        # Emissões de N2O (Wang et al., 2017)
        exposed_mass = 100  # kg (assumido para cálculo)
        exposed_hours = 8
        
        opening_factor = (exposed_mass / waste_kg_day) * (exposed_hours / 24)
        opening_factor = np.clip(opening_factor, 0.0, 1.0)
        
        E_open = 1.91  # g N/kg resíduo (frente de exposição)
        E_closed = 2.15 # g N/kg resíduo (área coberta)
        E_avg = opening_factor * E_open + (1 - opening_factor) * E_closed
        
        # Ajuste pela umidade
        moisture_factor = (1 - moisture_fraction) / (1 - 0.55)
        E_avg_adjusted = E_avg * moisture_factor
        
        # Emissão diária de N2O (convertida para kg)
        daily_n2o_kg = (E_avg_adjusted * (44/28) / 1_000_000) * waste_kg_day
        
        # Distribuição ao longo de 5 dias (perfil do aterro)
        kernel_n2o = np.array([self.profile_n2o_landfill.get(d, 0) for d in range(1, 6)], dtype=float)
        n2o_emissions = fftconvolve(np.full(days, daily_n2o_kg), kernel_n2o, mode='full')[:days]
        
        # Adicionar emissões de pré-descarte
        ch4_pre, n2o_pre = self._calculate_pre_disposal(waste_kg_day, days)
        
        return ch4_emissions + ch4_pre, n2o_emissions + n2o_pre
    
    def _calculate_pre_disposal(self, waste_kg_day, days):
        """Calcula as emissões de pré-descarte (antes da disposição final)"""
        ch4_emissions = np.full(days, waste_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o_emissions = np.zeros(days)
        
        for entry_day in range(days):
            for days_after, fraction in self.profile_n2o_pre.items():
                emission_day = entry_day + days_after - 1
                if emission_day < days:
                    n2o_emissions[emission_day] += (waste_kg_day * self.N2O_pre_kg_per_kg_day * fraction)
        
        return ch4_emissions, n2o_emissions
    
    def calculate_vermicomposting_emissions(self, waste_kg_day, moisture_fraction, years=20,
                                            f_ch4=None, f_n2o=None):
        """Calcula as emissões da vermicompostagem."""
        if f_ch4 is None:
            f_ch4 = self.f_CH4_vermi
        if f_n2o is None:
            f_n2o = self.f_N2O_vermi
        days = years * 365
        dry_fraction = 1 - moisture_fraction
        
        # Emissões por batelada (kg do gás)
        ch4_per_batch = (waste_kg_day * self.TOC * f_ch4 * (16/12) * dry_fraction)
        n2o_per_batch = (waste_kg_day * self.TN * f_n2o * (44/28) * dry_fraction)
        
        # Inicializar arrays
        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)
        
        # Distribuir as emissões ao longo do período de compostagem
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_per_batch * self.profile_ch4_vermi[compost_day]
                    n2o_emissions[emission_day] += n2o_per_batch * self.profile_n2o_vermi[compost_day]
        
        return ch4_emissions, n2o_emissions
    
    def calculate_thermophilic_emissions(self, waste_kg_day, moisture_fraction, years=20):
        """Calcula as emissões da compostagem termofílica (frações fixas)"""
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
    
    def calculate_avoided_emissions(self, waste_kg_day, k_year, temperature_C, 
                                    doc_fraction, moisture_fraction, years=20,
                                    gwp_ch4=None, gwp_n2o=None,
                                    f_ch4_vermi=None, f_n2o_vermi=None):
        """Calcula as emissões evitadas para ambas as tecnologias."""
        if gwp_ch4 is None:
            gwp_ch4 = self.GWP_CH4_20
        if gwp_n2o is None:
            gwp_n2o = self.GWP_N2O_20
        if f_ch4_vermi is None:
            f_ch4_vermi = self.f_CH4_vermi
        if f_n2o_vermi is None:
            f_n2o_vermi = self.f_N2O_vermi

        # Calcular todas as emissões
        ch4_landfill, n2o_landfill = self.calculate_landfill_emissions(
            waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years
        )
        
        ch4_vermi, n2o_vermi = self.calculate_vermicomposting_emissions(
            waste_kg_day, moisture_fraction, years,
            f_ch4=f_ch4_vermi, f_n2o=f_n2o_vermi
        )
        
        ch4_thermo, n2o_thermo = self.calculate_thermophilic_emissions(
            waste_kg_day, moisture_fraction, years
        )
        
        # Converter para CO2eq
        baseline_co2eq = (ch4_landfill * gwp_ch4 + n2o_landfill * gwp_n2o) / 1000
        vermi_co2eq = (ch4_vermi * gwp_ch4 + n2o_vermi * gwp_n2o) / 1000
        thermo_co2eq = (ch4_thermo * gwp_ch4 + n2o_thermo * gwp_n2o) / 1000
        
        # Emissões evitadas totais
        avoided_vermi = baseline_co2eq.sum() - vermi_co2eq.sum()
        avoided_thermo = baseline_co2eq.sum() - thermo_co2eq.sum()
        
        # Criar série de datas
        days = years * 365
        ano_inicio = datetime.now().year
        data_inicio = datetime(ano_inicio, 1, 1)
        datas = pd.date_range(start=data_inicio, periods=days, freq='D')
        
        # DataFrame diário detalhado
        df_detalhado = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_landfill,
            'N2O_Aterro_kg_dia': n2o_landfill,
            'CH4_Vermi_kg_dia': ch4_vermi,
            'N2O_Vermi_kg_dia': n2o_vermi,
            'CH4_Thermo_kg_dia': ch4_thermo,
            'N2O_Thermo_kg_dia': n2o_thermo,
        })
        
        # Adicionar colunas de CO2eq
        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi', 'CH4_Thermo', 'N2O_Thermo']:
            gwp = gwp_ch4 if 'CH4' in gas else gwp_n2o
            df_detalhado[f'{gas}_tCO2eq'] = df_detalhado[f'{gas}_kg_dia'] * gwp / 1000
        
        df_detalhado['Total_Aterro_tCO2eq_dia'] = df_detalhado['CH4_Aterro_tCO2eq'] + df_detalhado['N2O_Aterro_tCO2eq']
        df_detalhado['Total_Vermi_tCO2eq_dia'] = df_detalhado['CH4_Vermi_tCO2eq'] + df_detalhado['N2O_Vermi_tCO2eq']
        df_detalhado['Total_Thermo_tCO2eq_dia'] = df_detalhado['CH4_Thermo_tCO2eq'] + df_detalhado['N2O_Thermo_tCO2eq']
        
        df_detalhado['Total_Aterro_tCO2eq_acum'] = df_detalhado['Total_Aterro_tCO2eq_dia'].cumsum()
        df_detalhado['Total_Vermi_tCO2eq_acum'] = df_detalhado['Total_Vermi_tCO2eq_dia'].cumsum()
        df_detalhado['Total_Thermo_tCO2eq_acum'] = df_detalhado['Total_Thermo_tCO2eq_dia'].cumsum()
        
        df_detalhado['Reducao_Vermi_tCO2eq_acum'] = df_detalhado['Total_Aterro_tCO2eq_acum'] - df_detalhado['Total_Vermi_tCO2eq_acum']
        df_detalhado['Reducao_Thermo_tCO2eq_acum'] = df_detalhado['Total_Aterro_tCO2eq_acum'] - df_detalhado['Total_Thermo_tCO2eq_acum']
        
        df_detalhado['Ano'] = df_detalhado['Data'].dt.year
        
        # Resumo anual
        df_anual = df_detalhado.groupby('Ano').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
            'Total_Thermo_tCO2eq_dia': 'sum',
        }).reset_index()
        
        df_anual['Redução_Vermi_tCO2eq'] = df_anual['Total_Aterro_tCO2eq_dia'] - df_anual['Total_Vermi_tCO2eq_dia']
        df_anual['Redução_Thermo_tCO2eq'] = df_anual['Total_Aterro_tCO2eq_dia'] - df_anual['Total_Thermo_tCO2eq_dia']
        df_anual['Redução_Acumulada_Vermi_tCO2eq'] = df_anual['Redução_Vermi_tCO2eq'].cumsum()
        df_anual['Redução_Acumulada_Thermo_tCO2eq'] = df_anual['Redução_Thermo_tCO2eq'].cumsum()
        
        df_anual.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Emissões_Baseline_tCO2eq',
            'Total_Vermi_tCO2eq_dia': 'Emissões_Vermicompostagem_tCO2eq',
            'Total_Thermo_tCO2eq_dia': 'Emissões_Termofílica_tCO2eq',
        }, inplace=True)
        
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
                'superiority_percent': ((avoided_vermi / avoided_thermo) - 1) * 100
            },
            'annual_averages': {
                'baseline_tco2eq_year': baseline_co2eq.sum() / years,
                'vermi_avoided_year': avoided_vermi / years,
                'thermo_avoided_year': avoided_thermo / years
            },
            'detailed_data': {
                'daily': df_detalhado,
                'annual': df_anual
            }
        }
        
        return results

# =============================================================================
# FUNÇÕES AUXILIARES PARA ANÁLISE DE SENSIBILIDADE E MONTE CARLO
# =============================================================================
def run_sobol_sensitivity(calculator, waste_kg_day, moisture, years=20, n_samples=64):
    """Executa análise de sensibilidade de Sobol sobre emissões evitadas."""
    k_fixed = 0.06
    doc_fixed = 0.15

    def vermicomposting_model(params):
        T, U, fCH4, fN2O, GWP_CH4, GWP_N2O = params
        results = calculator.calculate_avoided_emissions(
            waste_kg_day, k_fixed, T, doc_fixed, U/100, years,
            gwp_ch4=GWP_CH4, gwp_n2o=GWP_N2O,
            f_ch4_vermi=fCH4, f_n2o_vermi=fN2O
        )
        return results['vermicomposting']['avoided_co2eq_t']

    def thermophilic_model(params):
        T, U, fCH4, fN2O, GWP_CH4, GWP_N2O = params
        results = calculator.calculate_avoided_emissions(
            waste_kg_day, k_fixed, T, doc_fixed, U/100, years,
            gwp_ch4=GWP_CH4, gwp_n2o=GWP_N2O
        )
        return results['thermophilic']['avoided_co2eq_t']

    problem = {
        'num_vars': 6,
        'names': ['T', 'U', 'fCH4', 'fN2O', 'GWP_CH4', 'GWP_N2O'],
        'bounds': [
            [20.0, 30.0],               # T (°C)
            [55.0, 85.0],                # U (%)
            [0.000107, 0.0013],          # fCH4
            [0.000739, 0.0092],          # fN2O
            [7.2, 79.7],                  # GWP_CH4
            [130.0, 273.0]                # GWP_N2O
        ]
    }

    param_values = sample(problem, n_samples, seed=50)
    results_vermi = Parallel(n_jobs=-1)(delayed(vermicomposting_model)(params) for params in param_values)
    results_thermo = Parallel(n_jobs=-1)(delayed(thermophilic_model)(params) for params in param_values)

    Si_vermi = analyze(problem, np.array(results_vermi), print_to_console=False)
    Si_thermo = analyze(problem, np.array(results_thermo), print_to_console=False)

    return {
        'vermi': Si_vermi,
        'thermo': Si_thermo,
        'problem': problem
    }

def run_monte_carlo_analysis(calculator, waste_kg_day, k, temp, doc, moisture, 
                            years=20, n_simulations=100,
                            prob_otimista=0.3, prob_real=0.5, prob_pessimista=0.2):
    """Executa análise de incerteza Monte Carlo."""
    np.random.seed(50)
    cenarios = {
        'otimista':  {'ch4': 79.7, 'n2o': 273},
        'real':      {'ch4': 27.0, 'n2o': 273},
        'pessimista':{'ch4': 7.2 , 'n2o': 130}
    }
    prob_list = [prob_otimista, prob_real, prob_pessimista]
    cenario_nomes = list(cenarios.keys())

    results_vermi = []
    results_thermo = []
    mc_parameters = []

    k_fixed = k
    doc_fixed = doc

    for i in range(n_simulations):
        T_mc = np.random.uniform(20.0, 30.0)
        U_mc = np.random.uniform(55.0, 85.0)
        fCH4_mc = np.random.uniform(0.000107, 0.0013)
        fN2O_mc = np.random.uniform(0.000739, 0.0092)

        cenario = np.random.choice(cenario_nomes, p=prob_list)
        gwp_ch4 = cenarios[cenario]['ch4']
        gwp_n2o = cenarios[cenario]['n2o']

        results = calculator.calculate_avoided_emissions(
            waste_kg_day, k_fixed, T_mc, doc_fixed, U_mc/100, years,
            gwp_ch4=gwp_ch4, gwp_n2o=gwp_n2o,
            f_ch4_vermi=fCH4_mc, f_n2o_vermi=fN2O_mc
        )

        results_vermi.append(results['vermicomposting']['avoided_co2eq_t'])
        results_thermo.append(results['thermophilic']['avoided_co2eq_t'])

        mc_parameters.append({
            'simulacao': i+1,
            'temperatura': T_mc,
            'umidade': U_mc,
            'fCH4': fCH4_mc,
            'fN2O': fN2O_mc,
            'cenario_gwp': cenario,
            'gwp_ch4': gwp_ch4,
            'gwp_n2o': gwp_n2o
        })

    mc_params_df = pd.DataFrame(mc_parameters)
    mc_params_df['vermi_evitadas'] = results_vermi
    mc_params_df['termo_evitadas'] = results_thermo

    return np.array(results_vermi), np.array(results_thermo), mc_params_df

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO (ADAPTADAS PARA STREAMLIT)
# =============================================================================
def create_dashboard(results, sensitivity, mc_vermi, mc_thermo, total_waste_tons, calculator):
    """Cria e retorna as figuras do painel principal."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Comparação de emissões evitadas
    ax = axes[0, 0]
    tecnologias = ['Vermicompostagem', 'Termofílica']
    evitadas = [results['vermicomposting']['avoided_co2eq_t'], 
                results['thermophilic']['avoided_co2eq_t']]
    
    bars = ax.bar(tecnologias, evitadas, color=['green', 'blue'])
    ax.set_ylabel('Emissões Evitadas (tCO₂eq)')
    ax.set_title('Emissões Evitadas em 20 anos')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, evitadas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(evitadas)*0.02,
                f'{val:.1f}', ha='center', va='bottom')
    
    # 2. Detalhamento das emissões por fonte
    ax = axes[0, 1]
    categorias = ['Baseline\nCH₄', 'Baseline\nN₂O', 'Vermi\nCH₄', 'Vermi\nN₂O', 
                  'Termo\nCH₄', 'Termo\nN₂O']
    emissoes = [
        results['baseline']['ch4_kg'] * calculator.GWP_CH4_20 / 1000,
        results['baseline']['n2o_kg'] * calculator.GWP_N2O_20 / 1000,
        results['vermicomposting']['ch4_kg'] * calculator.GWP_CH4_20 / 1000,
        results['vermicomposting']['n2o_kg'] * calculator.GWP_N2O_20 / 1000,
        results['thermophilic']['ch4_kg'] * calculator.GWP_CH4_20 / 1000,
        results['thermophilic']['n2o_kg'] * calculator.GWP_N2O_20 / 1000
    ]
    
    cores = ['red', 'darkred', 'green', 'darkgreen', 'blue', 'darkblue']
    bars = ax.bar(categorias, emissoes, color=cores)
    ax.set_ylabel('Emissões (tCO₂eq)')
    ax.set_title('Detalhamento das Emissões por Fonte')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3. Índices de sensibilidade
    ax = axes[0, 2]
    parametros = ['T', 'U', 'fCH₄', 'fN₂O', 'GWP_CH₄', 'GWP_N₂O']
    x = np.arange(len(parametros))
    width = 0.35
    
    ax.bar(x - width/2, sensitivity['vermi']['ST'], width, label='Vermicompostagem', alpha=0.8)
    ax.bar(x + width/2, sensitivity['thermo']['ST'], width, label='Termofílica', alpha=0.8)
    
    ax.set_xlabel('Parâmetro')
    ax.set_ylabel('Índice de Sensibilidade Total (ST)')
    ax.set_title('Análise de Sensibilidade Global')
    ax.set_xticks(x)
    ax.set_xticklabels(parametros, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribuições Monte Carlo
    ax = axes[1, 0]
    sns.histplot(mc_vermi, kde=True, color='green', label='Vermicompostagem', alpha=0.6, ax=ax)
    sns.histplot(mc_thermo, kde=True, color='blue', label='Termofílica', alpha=0.6, ax=ax)
    
    ax.axvline(results['vermicomposting']['avoided_co2eq_t'], color='green', linestyle='--',
               label=f'Média Vermi: {results["vermicomposting"]["avoided_co2eq_t"]:.1f}')
    ax.axvline(results['thermophilic']['avoided_co2eq_t'], color='blue', linestyle='--',
               label=f'Média Termo: {results["thermophilic"]["avoided_co2eq_t"]:.1f}')
    
    ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
    ax.set_ylabel('Frequência')
    ax.set_title('Análise de Incerteza Monte Carlo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Fatores de emissão por tonelada de resíduo
    ax = axes[1, 1]
    fator_vermi = results['vermicomposting']['avoided_co2eq_t'] / total_waste_tons
    fator_thermo = results['thermophilic']['avoided_co2eq_t'] / total_waste_tons
    
    fatores = [fator_vermi, fator_thermo]
    tecnologias = ['Vermicompostagem', 'Termofílica']
    
    bars = ax.bar(tecnologias, fatores, color=['green', 'blue'])
    ax.set_ylabel('Emissões Evitadas (tCO₂eq/t residuo)')
    ax.set_title('Fatores de Emissão por Tonelada de Resíduo')
    ax.grid(True, alpha=0.3)
    
    for bar, fator in zip(bars, fatores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fatores)*0.02,
                f'{fator:.3f}', ha='center', va='bottom')
    
    # 6. Métricas resumidas
    ax = axes[1, 2]
    ax.axis('off')
    
    texto_resumo = f"""
    RESUMO DAS MÉTRICAS
    
    Total de Resíduos: {total_waste_tons:.0f} t
    
    VERMICOMPOSTAGEM:
    • Evitado: {results['vermicomposting']['avoided_co2eq_t']:.1f} tCO₂eq
    • Anual: {results['annual_averages']['vermi_avoided_year']:.2f} tCO₂eq/ano
    • Por ton: {fator_vermi:.3f} tCO₂eq/t
    
    TERMOFÍLICA:
    • Evitado: {results['thermophilic']['avoided_co2eq_t']:.1f} tCO₂eq
    • Anual: {results['annual_averages']['thermo_avoided_year']:.2f} tCO₂eq/ano
    • Por ton: {fator_thermo:.3f} tCO₂eq/t
    
    COMPARAÇÃO:
    • Diferença: {results['comparison']['difference_tco2eq']:.2f} tCO₂eq
    • Superioridade: {results['comparison']['superiority_percent']:.1f}%
    
    SENSIBILIDADE (Vermi ST):
    • T: {sensitivity['vermi']['ST'][0]:.3f}
    • U: {sensitivity['vermi']['ST'][1]:.3f}
    • fCH₄: {sensitivity['vermi']['ST'][2]:.3f}
    • fN₂O: {sensitivity['vermi']['ST'][3]:.3f}
    • GWP_CH₄: {sensitivity['vermi']['ST'][4]:.3f}
    • GWP_N₂O: {sensitivity['vermi']['ST'][5]:.3f}
    """
    
    ax.text(0.05, 0.95, texto_resumo, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    plt.suptitle('Resultados da Análise de Emissões de GEE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_emissions_accumulated_plot(results):
    """Gráfico de emissões acumuladas ao longo do tempo."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_detalhado = results['detailed_data']['daily']
    ax.plot(df_detalhado['Data'], df_detalhado['Total_Aterro_tCO2eq_acum'], 
            label='Baseline (Aterro)', color='red', linewidth=2)
    ax.plot(df_detalhado['Data'], df_detalhado['Total_Vermi_tCO2eq_acum'], 
            label='Vermicompostagem', color='green', linewidth=2)
    ax.plot(df_detalhado['Data'], df_detalhado['Total_Thermo_tCO2eq_acum'], 
            label='Termofílica', color='blue', linewidth=2)
    ax.set_xlabel('Data')
    ax.set_ylabel('Emissões Acumuladas (tCO₂eq)')
    ax.set_title('Emissões Acumuladas de GEE ao Longo do Tempo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def create_annual_emissions_plot(results):
    """Gráfico de barras anuais."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_anual = results['detailed_data']['annual']
    x = np.arange(len(df_anual))
    width = 0.25
    
    ax.bar(x - width, df_anual['Emissões_Baseline_tCO2eq'], width, label='Baseline', color='red')
    ax.bar(x, df_anual['Emissões_Vermicompostagem_tCO2eq'], width, label='Vermicompostagem', color='green')
    ax.bar(x + width, df_anual['Emissões_Termofílica_tCO2eq'], width, label='Termofílica', color='blue')
    
    ax.set_xlabel('Ano')
    ax.set_ylabel('Emissões Anuais (tCO₂eq)')
    ax.set_title('Emissões Anuais de GEE por Tecnologia')
    ax.set_xticks(x)
    ax.set_xticklabels(df_anual['Ano'].astype(str), rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_tornado_plot(sensitivity):
    """Gráfico de tornado para sensibilidade."""
    fig, ax = plt.subplots(figsize=(10, 6))
    params = ['T', 'U', 'fCH4', 'fN2O', 'GWP_CH4', 'GWP_N2O']
    vermi_st = sensitivity['vermi']['ST']
    thermo_st = sensitivity['thermo']['ST']
    
    y_pos = np.arange(len(params))
    
    ax.barh(y_pos - 0.2, vermi_st, height=0.4, label='Vermicompostagem', color='green', alpha=0.7)
    ax.barh(y_pos + 0.2, thermo_st, height=0.4, label='Termofílica', color='blue', alpha=0.7)
    
    ax.set_xlabel('Índice de Sensibilidade Total (ST)')
    ax.set_title('Comparação da Sensibilidade dos Parâmetros')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig

# =============================================================================
# FUNÇÃO PRINCIPAL STREAMLIT
# =============================================================================
def main():
    st.set_page_config(
        page_title="Nutriwash Circular System BR",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado para formatação brasileira e estética
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #2e7d32;
            color: white;
        }
        .stButton>button:hover {
            background-color: #1b5e20;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("♻️ Nutriwash Circular System BR")
    st.markdown("### Calculadora de Emissões de GEE para Gestão de Resíduos")
    st.markdown("---")
    
    # Sidebar - Parâmetros de entrada
    with st.sidebar:
        st.header("📊 Parâmetros de Entrada")
        st.markdown("**Dados de Resíduos**")
        waste_kg_day = st.number_input("Resíduos diários (kg/dia)", min_value=1, max_value=100000, value=100, step=10)
        years = st.number_input("Período de simulação (anos)", min_value=1, max_value=50, value=20, step=1)
        
        st.markdown("**Parâmetros do Aterro**")
        k_year = st.number_input("Taxa de decaimento k (ano⁻¹)", min_value=0.01, max_value=0.20, value=0.06, step=0.01, format="%.3f")
        temperature = st.number_input("Temperatura do aterro (°C)", min_value=10, max_value=40, value=25, step=1)
        doc_fraction = st.number_input("Fração DOC", min_value=0.05, max_value=0.30, value=0.15, step=0.01, format="%.3f")
        
        st.markdown("**Umidade do Resíduo**")
        moisture = st.number_input("Umidade (%)", min_value=50, max_value=95, value=85, step=5) / 100.0
        
        st.markdown("**Cenários de GWP (Monte Carlo)**")
        prob_otimista = st.slider("Probabilidade Cenário Otimista (GWP-20)", 0.0, 1.0, 0.3, step=0.05)
        prob_real = st.slider("Probabilidade Cenário Realista (GWP-100)", 0.0, 1.0, 0.5, step=0.05)
        prob_pessimista = st.slider("Probabilidade Cenário Pessimista (GWP-500)", 0.0, 1.0, 0.2, step=0.05)
        
        # Garantir soma 1
        total_prob = prob_otimista + prob_real + prob_pessimista
        if abs(total_prob - 1.0) > 0.01:
            st.warning(f"As probabilidades somam {total_prob:.2f}. Ajustando para soma 1.")
            prob_otimista /= total_prob
            prob_real /= total_prob
            prob_pessimista /= total_prob
        
        run_button = st.button("▶️ Executar Análise", use_container_width=True)
    
    if run_button:
        with st.spinner("Processando dados e executando cálculos..."):
            # Instanciar calculadora
            calculator = GHGEmissionCalculator()
            
            # Calcular resultados determinísticos para cenário realista (padrão)
            results = calculator.calculate_avoided_emissions(
                waste_kg_day, k_year, temperature, doc_fraction, moisture, years,
                gwp_ch4=27.0, gwp_n2o=273  # cenário realista
            )
            
            # Análise de sensibilidade
            st.info("Executando análise de sensibilidade de Sobol...")
            sensitivity = run_sobol_sensitivity(calculator, waste_kg_day, moisture, years)
            
            # Monte Carlo
            st.info("Executando simulação Monte Carlo...")
            mc_vermi, mc_thermo, mc_params = run_monte_carlo_analysis(
                calculator, waste_kg_day, k_year, temperature, doc_fraction, moisture, years,
                n_simulations=100,
                prob_otimista=prob_otimista, prob_real=prob_real, prob_pessimista=prob_pessimista
            )
            
            total_waste_tons = waste_kg_day * 365 * years / 1000
            
            # Exibir métricas principais
            st.subheader("📈 Resultados Principais")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emissões Evitadas - Vermicompostagem", 
                          f"{results['vermicomposting']['avoided_co2eq_t']:,.1f} tCO₂eq".replace(',', 'X').replace('.', ',').replace('X', '.'))
            with col2:
                st.metric("Emissões Evitadas - Termofílica", 
                          f"{results['thermophilic']['avoided_co2eq_t']:,.1f} tCO₂eq".replace(',', 'X').replace('.', ',').replace('X', '.'))
            with col3:
                diff = results['comparison']['difference_tco2eq']
                sup = results['comparison']['superiority_percent']
                st.metric("Superioridade Vermicompostagem", 
                          f"{sup:.1f}%", delta=f"{diff:+.1f} tCO₂eq")
            
            # Painel principal
            st.subheader("📊 Painel de Resultados")
            fig_dashboard = create_dashboard(results, sensitivity, mc_vermi, mc_thermo, total_waste_tons, calculator)
            st.pyplot(fig_dashboard)
            
            # Gráficos adicionais em abas
            tab1, tab2, tab3 = st.tabs(["📈 Emissões Acumuladas", "📅 Emissões Anuais", "🎲 Análise de Sensibilidade"])
            with tab1:
                fig_acum = create_emissions_accumulated_plot(results)
                st.pyplot(fig_acum)
            with tab2:
                fig_annual = create_annual_emissions_plot(results)
                st.pyplot(fig_annual)
            with tab3:
                fig_tornado = create_tornado_plot(sensitivity)
                st.pyplot(fig_tornado)
            
            # Exibir estatísticas do Monte Carlo
            st.subheader("🎲 Estatísticas da Simulação Monte Carlo")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Vermicompostagem**")
                st.write(f"Média: {np.mean(mc_vermi):.1f} tCO₂eq")
                st.write(f"Desvio Padrão: {np.std(mc_vermi):.1f} tCO₂eq")
                st.write(f"Percentil 5: {np.percentile(mc_vermi, 5):.1f} tCO₂eq")
                st.write(f"Percentil 95: {np.percentile(mc_vermi, 95):.1f} tCO₂eq")
            with col2:
                st.markdown("**Termofílica**")
                st.write(f"Média: {np.mean(mc_thermo):.1f} tCO₂eq")
                st.write(f"Desvio Padrão: {np.std(mc_thermo):.1f} tCO₂eq")
                st.write(f"Percentil 5: {np.percentile(mc_thermo, 5):.1f} tCO₂eq")
                st.write(f"Percentil 95: {np.percentile(mc_thermo, 95):.1f} tCO₂eq")
            
            prob_vermi_melhor = np.mean(mc_vermi > mc_thermo) * 100
            st.success(f"✅ Probabilidade de a vermicompostagem superar a termofílica: **{prob_vermi_melhor:.1f}%**")
            
            # Opção de download dos dados
            st.subheader("📥 Exportar Dados")
            df_daily = results['detailed_data']['daily']
            df_annual = results['detailed_data']['annual']
            
            col1, col2 = st.columns(2)
            with col1:
                csv_daily = df_daily.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar dados diários (CSV)", csv_daily, "emissoes_diarias.csv", "text/csv")
            with col2:
                csv_annual = df_annual.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar dados anuais (CSV)", csv_annual, "emissoes_anuais.csv", "text/csv")
    
    else:
        st.info("👈 Configure os parâmetros na barra lateral e clique em 'Executar Análise' para começar.")
        st.markdown("""
        ### Sobre o Aplicativo
        Este aplicativo calcula as emissões de Gases de Efeito Estufa (GEE) para três cenários de gestão de resíduos:
        - **Baseline**: Aterro Sanitário (metodologia IPCC 2006)
        - **Tecnologia 1**: Vermicompostagem com minhocas em reatores
        - **Tecnologia 2**: Compostagem Termofílica em leiras abertas
        
        **Funcionalidades**:
        - Análise determinística das emissões evitadas
        - Análise de sensibilidade global (Sobol) com 6 parâmetros
        - Simulação de incerteza Monte Carlo com cenários de GWP (20, 100 e 500 anos)
        - Visualizações interativas e exportação de dados
        
        **Metodologias**: IPCC (2006), Yang et al. (2017), Feng et al. (2020), Wang et al. (2017)
        """)

if __name__ == "__main__":
    main()
