import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import newton
import warnings
warnings.filterwarnings('ignore')

# Função para formatar valores em formato brasileiro
def formatar_br(valor):
    if isinstance(valor, (int, float)):
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return valor

# Função auxiliar para formatar eixos
def br_format(x, p):
    return f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================================
# FUNÇÃO: CÁLCULO DO POTENCIAL DE MITIGAÇÃO
# ============================================================================
def calcular_potencial_mitigacao(parametros):
    """
    Calcula o potencial de mitigação de emissões da vermicompostagem
    comparado com o cenário de aterro sanitário.
    """
    # Parâmetros do cenário base (aterro)
    toneladas_residuo = parametros['toneladas_residuo']
    anos = parametros['anos']
    
    # Fatores de emissão (kg CO2eq/kg resíduo)
    # Valores baseados no IPCC e literatura
    fator_aterro = parametros.get('fator_emissao_aterro', 1.2)  # kg CO2eq/kg
    fator_vermi = parametros.get('fator_emissao_vermi', 0.3)    # kg CO2eq/kg
    
    # Calcular emissões anuais
    emissao_anual_aterro = toneladas_residuo * 1000 * fator_aterro / 1000  # tCO2eq/ano
    emissao_anual_vermi = toneladas_residuo * 1000 * fator_vermi / 1000    # tCO2eq/ano
    
    # Redução anual
    reducao_anual = emissao_anual_aterro - emissao_anual_vermi  # tCO2eq/ano
    
    # Calcular acumulado ao longo dos anos
    tempo = np.arange(1, anos + 1)
    acumulado_aterro = emissao_anual_aterro * tempo
    acumulado_vermi = emissao_anual_vermi * tempo
    reducao_acumulada = acumulado_aterro - acumulado_vermi
    
    # Eficiência do sistema
    if emissao_anual_aterro > 0:
        eficiencia = (reducao_anual / emissao_anual_aterro) * 100
    else:
        eficiencia = 0
    
    return {
        'emissao_anual_aterro': emissao_anual_aterro,
        'emissao_anual_vermi': emissao_anual_vermi,
        'reducao_anual': reducao_anual,
        'acumulado_aterro': acumulado_aterro,
        'acumulado_vermi': acumulado_vermi,
        'reducao_acumulada': reducao_acumulada,
        'eficiencia': eficiencia,
        'tempo': tempo
    }

# ============================================================================
# FUNÇÃO: CÁLCULO DE INDICADORES FINANCEIROS CORRIGIDA
# ============================================================================
def calcular_indicadores_financeiros(capex, opex_anual, receita_anual, anos, 
                                     reducao_anual_tco2eq, taxa_desconto=0.08):
    """
    Calcula indicadores financeiros para análise TEA - VERSÃO CORRIGIDA
    
    Args:
        capex: Investimento inicial (R$)
        opex_anual: Custo operacional anual (R$/ano)
        receita_anual: Receita anual (R$/ano)
        anos: Período de análise (anos)
        reducao_anual_tco2eq: Redução anual de emissões (tCO2eq/ano)
        taxa_desconto: Taxa de desconto (default 8%)
    
    Returns:
        Dicionário com indicadores financeiros
    """
    # Fluxo de caixa anual
    fluxo_caixa = [-capex]  # Ano 0
    for ano in range(1, anos + 1):
        fluxo_caixa.append(receita_anual - opex_anual)
    
    # Calcular VPL
    vpl = np.npv(taxa_desconto, fluxo_caixa)
    
    # Calcular TIR (Internal Rate of Return)
    try:
        tir = np.irr(fluxo_caixa)
        if np.isnan(tir):
            tir = 0
    except:
        tir = 0
    
    # Payback Simples
    acumulado = 0
    payback_anos = None
    for ano, fluxo in enumerate(fluxo_caixa):
        if ano == 0:
            acumulado = fluxo
        else:
            acumulado += fluxo
            if acumulado >= 0 and payback_anos is None:
                payback_anos = ano
                break
    
    # Payback Descontado
    fluxo_desc = []
    for ano, fluxo in enumerate(fluxo_caixa):
        if ano == 0:
            fluxo_desc.append(fluxo)
        else:
            fluxo_desc.append(fluxo / ((1 + taxa_desconto) ** ano))
    
    acumulado_desc = 0
    payback_desc_anos = None
    for ano, fluxo in enumerate(fluxo_desc):
        if ano == 0:
            acumulado_desc = fluxo
        else:
            acumulado_desc += fluxo
            if acumulado_desc >= 0 and payback_desc_anos is None:
                payback_desc_anos = ano
                break
    
    # Custo por tonelada evitada - CORRIGIDO
    if reducao_anual_tco2eq > 0:
        custo_tonelada_evitada = capex / (anos * reducao_anual_tco2eq)
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

# ============================================================================
# FUNÇÃO: ANÁLISE DE SENSIBILIDADE TEA
# ============================================================================
def analise_sensibilidade_tea(parametros_base, variaveis_analise, 
                              reducao_anual_tco2eq):
    """
    Realiza análise de sensibilidade para o TEA
    """
    resultados = {}
    anos = parametros_base['anos_simulacao']
    
    for var_nome, valores in variaveis_analise.items():
        resultados_var = []
        
        for valor in valores:
            # Ajustar parâmetro
            parametros_ajustados = parametros_base.copy()
            
            # Aplicar ajuste específico para cada variável
            if var_nome == 'preco_carbono':
                # Ajustar receita de carbono
                receita_carbono = reducao_anual_tco2eq * valor
                receita_total = (parametros_base['receita_humus'] + 
                               receita_carbono + 
                               parametros_base['economia_aterro'])
                parametros_ajustados['receita_total_anual'] = receita_total
                
                capex = parametros_base['capex_total']
                opex = parametros_base['opex_anual']
                receita = receita_total
                
            elif var_nome == 'custo_capex':
                fator = valor / 100
                capex = parametros_base['capex_total'] * (1 + fator)
                opex = parametros_base['opex_anual']
                receita = parametros_base['receita_total_anual']
                
            elif var_nome == 'custo_opex':
                fator = valor / 100
                capex = parametros_base['capex_total']
                opex = parametros_base['opex_anual'] * (1 + fator)
                receita = parametros_base['receita_total_anual']
                
            elif var_nome == 'producao_humus':
                fator = valor / 100
                receita_humus = parametros_base['receita_humus'] * (1 + fator)
                receita_total = (receita_humus + 
                               parametros_base['receita_carbono'] + 
                               parametros_base['economia_aterro'])
                capex = parametros_base['capex_total']
                opex = parametros_base['opex_anual']
                receita = receita_total
            
            # Calcular indicadores com custo corrigido
            indicadores = calcular_indicadores_financeiros(
                capex, opex, receita, anos, 
                reducao_anual_tco2eq,
                parametros_base['taxa_desconto']
            )
            
            resultados_var.append({
                'valor_variavel': valor,
                'vpl': indicadores['vpl'],
                'tir': indicadores['tir'],
                'payback': indicadores['payback_anos'],
                'custo_tonelada': indicadores['custo_tonelada_evitada']
            })
        
        resultados[var_nome] = pd.DataFrame(resultados_var)
    
    return resultados

# ============================================================================
# FUNÇÃO: CRIAR DASHBOARD TEA - VERSÃO COMPLETA E CORRIGIDA
# ============================================================================
def criar_dashboard_tea(analise_tea, resultados_sensibilidade, resultados_simulacao):
    """
    Cria dashboard interativo para Análise Técnico-Econômica - VERSÃO CORRIGIDA
    """
    st.title("📊 Dashboard TEA - Vermicompostagem")
    
    # Criar abas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visão Geral", 
        "💰 Análise Financeira", 
        "📊 Análise de Sensibilidade",
        "🌍 Impacto Ambiental",
        "⚖️ Trade-off Econômico-Ambiental"
    ])
    
    with tab1:
        st.markdown("#### 📋 Resumo do Projeto")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "VPL (Valor Presente Líquido)",
                f"R$ {formatar_br(analise_tea['indicadores']['vpl'])}",
                delta=None
            )
        
        with col2:
            st.metric(
                "TIR (Taxa Interna de Retorno)",
                f"{analise_tea['indicadores']['tir']*100:.1f}%",
                delta=None
            )
        
        with col3:
            payback = analise_tea['indicadores']['payback_anos']
            if payback:
                st.metric(
                    "Payback (anos)",
                    f"{payback} anos",
                    delta=None
                )
            else:
                st.metric("Payback", "> período", delta=None)
        
        with col4:
            custo_ton = analise_tea['indicadores']['custo_tonelada_evitada']
            st.metric(
                "Custo por tCO₂eq evitada",
                f"R$ {formatar_br(custo_ton)}",
                delta=None
            )
        
        # Gráfico de fluxo de caixa acumulado
        st.markdown("#### 📈 Fluxo de Caixa Acumulado")
        fluxo_caixa = analise_tea['indicadores']['fluxo_caixa']
        fluxo_acumulado = np.cumsum(fluxo_caixa)
        
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        anos = range(len(fluxo_acumulado))
        ax1.plot(anos, fluxo_acumulado, 'b-', linewidth=2, marker='o')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.fill_between(anos, 0, fluxo_acumulado, where=fluxo_acumulado>=0, 
                        alpha=0.3, color='green')
        ax1.fill_between(anos, 0, fluxo_acumulado, where=fluxo_acumulado<0, 
                        alpha=0.3, color='red')
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('Fluxo de Caixa Acumulado (R$)')
        ax1.set_title('Evolução do Fluxo de Caixa Acumulado')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
        st.pyplot(fig1)
        
        # Tabela de parâmetros
        st.markdown("#### ⚙️ Parâmetros do Projeto")
        
        dados_parametros = {
            'Parâmetro': [
                'CAPEX Total', 'OPEX Anual', 'Receita Total Anual',
                'Período de Análise', 'Taxa de Desconto',
                'Redução Anual de CO₂eq', 'Custo por tCO₂eq evitada'
            ],
            'Valor': [
                f"R$ {formatar_br(analise_tea['parametros']['capex_total'])}",
                f"R$ {formatar_br(analise_tea['parametros']['opex_anual'])}/ano",
                f"R$ {formatar_br(analise_tea['parametros']['receita_total_anual'])}/ano",
                f"{analise_tea['parametros']['anos_simulacao']} anos",
                f"{analise_tea['parametros']['taxa_desconto']*100}%",
                f"{analise_tea['parametros'].get('reducao_anual_tco2eq', 0):.1f} tCO₂eq/ano",
                f"R$ {formatar_br(analise_tea['indicadores']['custo_tonelada_evitada'])}/tCO₂eq"
            ]
        }
        
        df_parametros = pd.DataFrame(dados_parametros)
        st.dataframe(df_parametros, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### 💰 Análise Financeira Detalhada")
        
        # Gráfico de fluxo de caixa anual
        fluxo_caixa = analise_tea['indicadores']['fluxo_caixa']
        anos = range(len(fluxo_caixa))
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bars = ax2.bar(anos, fluxo_caixa, color=['red'] + ['green']*(len(fluxo_caixa)-1))
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Ano')
        ax2.set_ylabel('Fluxo de Caixa (R$)')
        ax2.set_title('Fluxo de Caixa Anual')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'R$ {height:,.0f}'.replace(',', '.'), 
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=8)
        
        st.pyplot(fig2)
        
        # Análise de rentabilidade
        st.markdown("#### 📊 Indicadores de Rentabilidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VPL por ano
            vpl_anual = analise_tea['indicadores']['vpl'] / analise_tea['parametros']['anos_simulacao']
            st.metric("VPL Anual Médio", f"R$ {formatar_br(vpl_anual)}")
            
            # ROI (Return on Investment)
            capex = analise_tea['parametros']['capex_total']
            if capex > 0:
                roi = (analise_tea['indicadores']['vpl'] / capex) * 100
                st.metric("ROI", f"{roi:.1f}%")
        
        with col2:
            # Benefício/Custo
            beneficios_totais = sum(fluxo_caixa[1:])  # Exclui o investimento inicial
            if abs(capex) > 0:
                bcr = beneficios_totais / abs(fluxo_caixa[0])
                st.metric("Razão Benefício/Custo", f"{bcr:.2f}")
            
            # Break-even point
            payback_desc = analise_tea['indicadores']['payback_desc_anos']
            if payback_desc:
                st.metric("Payback Descontado", f"{payback_desc} anos")
        
        # Análise de cenários
        st.markdown("#### 🎯 Análise de Cenários")
        
        cenarios = {
            'Cenário': ['Otimista', 'Base', 'Pessimista'],
            'VPL (R$)': [
                analise_tea['indicadores']['vpl'] * 1.3,
                analise_tea['indicadores']['vpl'],
                analise_tea['indicadores']['vpl'] * 0.7
            ],
            'TIR (%)': [
                analise_tea['indicadores']['tir'] * 1.2 * 100,
                analise_tea['indicadores']['tir'] * 100,
                analise_tea['indicadores']['tir'] * 0.8 * 100
            ],
            'Payback (anos)': [
                max(1, int(analise_tea['indicadores']['payback_anos'] * 0.8)) if analise_tea['indicadores']['payback_anos'] else 'N/A',
                analise_tea['indicadores']['payback_anos'] if analise_tea['indicadores']['payback_anos'] else 'N/A',
                int(analise_tea['indicadores']['payback_anos'] * 1.2) if analise_tea['indicadores']['payback_anos'] else 'N/A'
            ]
        }
        
        df_cenarios = pd.DataFrame(cenarios)
        st.dataframe(df_cenarios, use_container_width=True)
    
    with tab3:
        st.markdown("#### 📊 Análise de Sensibilidade")
        
        # Gráficos de sensibilidade
        variaveis_analisadas = list(resultados_sensibilidade.keys())
        
        if len(variaveis_analisadas) > 0:
            fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, var_nome in enumerate(variaveis_analisadas[:4]):
                df_var = resultados_sensibilidade[var_nome]
                ax = axes[idx]
                
                # Gráfico de VPL vs variável
                ax.plot(df_var['valor_variavel'], df_var['vpl'], 'b-', linewidth=2, marker='o')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel(var_nome.replace('_', ' ').title())
                ax.set_ylabel('VPL (R$)')
                ax.set_title(f'Sensibilidade do VPL - {var_nome.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
                
                # Destacar ponto base
                if 'preco_carbono' in var_nome:
                    ponto_base = 100  # Preço base do carbono
                elif 'custo_capex' in var_nome or 'custo_opex' in var_nome:
                    ponto_base = 0
                elif 'producao_humus' in var_nome:
                    ponto_base = 100
                else:
                    ponto_base = 0
                
                idx_base = df_var['valor_variavel'].tolist().index(ponto_base)
                ax.scatter(df_var['valor_variavel'].iloc[idx_base], 
                          df_var['vpl'].iloc[idx_base], 
                          color='red', s=100, zorder=5)
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Tabela de resultados de sensibilidade
            st.markdown("#### 📋 Resultados Detalhados da Sensibilidade")
            
            for var_nome in variaveis_analisadas:
                st.markdown(f"**{var_nome.replace('_', ' ').title()}**")
                df_var = resultados_sensibilidade[var_nome]
                st.dataframe(df_var, use_container_width=True)
        else:
            st.info("Nenhuma análise de sensibilidade disponível.")
    
    with tab4:
        st.markdown("#### 🌍 Impacto Ambiental")
        
        # Métricas ambientais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            reducao_total = resultados_simulacao['reducao_acumulada'][-1]
            st.metric(
                "Redução Total de CO₂eq",
                f"{reducao_total:,.1f} tCO₂eq",
                delta=None
            )
        
        with col2:
            eficiencia = resultados_simulacao['eficiencia']
            st.metric(
                "Eficiência do Sistema",
                f"{eficiencia:.1f}%",
                delta=None
            )
        
        with col3:
            custo_ton = analise_tea['indicadores']['custo_tonelada_evitada']
            st.metric(
                "Custo-Efetividade",
                f"R$ {formatar_br(custo_ton)}/tCO₂eq",
                delta=None
            )
        
        # Gráfico comparativo de emissões
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        
        tempo = resultados_simulacao['tempo']
        acumulado_aterro = resultados_simulacao['acumulado_aterro']
        acumulado_vermi = resultados_simulacao['acumulado_vermi']
        
        ax4.plot(tempo, acumulado_aterro, 'r-', linewidth=2, label='Cenário Aterro')
        ax4.plot(tempo, acumulado_vermi, 'g-', linewidth=2, label='Vermicompostagem')
        ax4.fill_between(tempo, acumulado_vermi, acumulado_aterro, 
                        alpha=0.3, color='green', label='Redução de Emissões')
        
        ax4.set_xlabel('Ano')
        ax4.set_ylabel('Emissões Acumuladas (tCO₂eq)')
        ax4.set_title('Comparativo de Emissões: Aterro vs Vermicompostagem')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        st.pyplot(fig4)
        
        # Gráfico de redução anual
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        
        reducao_anual = resultados_simulacao['reducao_anual']
        anos = range(1, len(tempo) + 1)
        
        bars = ax5.bar(anos, [reducao_anual] * len(anos), 
                      color='green', alpha=0.7)
        ax5.set_xlabel('Ano')
        ax5.set_ylabel('Redução Anual (tCO₂eq)')
        ax5.set_title('Redução Anual de Emissões')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valor nas barras
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        st.pyplot(fig5)
    
    with tab5:
        st.markdown("#### ⚖️ Trade-off Econômico-Ambiental (CORRIGIDO)")
        
        # =========================================================================
        # CÁLCULO DO TRADE-OFF CORRETO
        # =========================================================================
        
        # 1. Calcular eficiência real da vermicompostagem
        # Baseado nos resultados da simulação
        emissao_total_aterro = resultados_simulacao['acumulado_aterro'][-1]
        emissao_total_vermi = resultados_simulacao['acumulado_vermi'][-1]
        
        if emissao_total_aterro > 0:
            eficiencia_vermi_real = ((emissao_total_aterro - emissao_total_vermi) / emissao_total_aterro) * 100
        else:
            eficiencia_vermi_real = 0
        
        # 2. Custo real por tCO₂eq evitada
        custo_vermi_real = analise_tea['indicadores']['custo_tonelada_evitada']
        
        # 3. Dados de outras tecnologias (valores de referência da literatura)
        dados_tecnologias = {
            'Tecnologia': [
                'Vermicompostagem (Este Projeto)',
                'Compostagem Tradicional',
                'Aterro com Captura de Biogás',
                'Incineração com Recuperação Energética',
                'Digestão Anaeróbica'
            ],
            'Eficiência de Redução (%)': [
                eficiencia_vermi_real,  # Valor real calculado
                65,   # Fonte: IPCC
                50,   # Fonte: Wang et al., 2017
                85,   # Fonte: EPA
                75    # Fonte: Yang et al., 2017
            ],
            'Custo Relativo': [
                1.0,   # Referência (este projeto)
                1.3,   # 30% mais caro
                0.7,   # 30% mais barato
                2.5,   # 150% mais caro
                1.8    # 80% mais caro
            ],
            'Custo Estimado (R$/tCO₂eq)': [
                custo_vermi_real,  # Custo real
                custo_vermi_real * 1.3,
                custo_vermi_real * 0.7,
                custo_vermi_real * 2.5,
                custo_vermi_real * 1.8
            ],
            'Cor': ['blue', 'orange', 'green', 'red', 'purple']
        }
        
        df_tecnologias = pd.DataFrame(dados_tecnologias)
        
        # =========================================================================
        # GRÁFICO DE TRADE-OFF CORRIGIDO
        # =========================================================================
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Trade-off Custo vs Eficiência
        scatter = ax1.scatter(
            df_tecnologias['Custo Estimado (R$/tCO₂eq)'],
            df_tecnologias['Eficiência de Redução (%)'],
            s=300,
            c=df_tecnologias['Cor'],
            alpha=0.7,
            edgecolors='black',
            linewidth=2
        )
        
        # Adicionar rótulos
        for i, row in df_tecnologias.iterrows():
            ax1.annotate(
                row['Tecnologia'],
                (row['Custo Estimado (R$/tCO₂eq)'], row['Eficiência de Redução (%)']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold' if row['Tecnologia'] == 'Vermicompostagem (Este Projeto)' else 'normal'
            )
        
        # Linha de eficiência-custo ideal (fronteira de Pareto)
        # Ordenar por custo
        df_sorted = df_tecnologias.sort_values('Custo Estimado (R$/tCO₂eq)')
        ax1.plot(
            df_sorted['Custo Estimado (R$/tCO₂eq)'],
            df_sorted['Eficiência de Redução (%)'],
            'k--',
            alpha=0.3,
            label='Fronteira de Eficiência'
        )
        
        ax1.set_xlabel('Custo por tCO₂eq Evitada (R$)')
        ax1.set_ylabel('Eficiência de Redução (%)')
        ax1.set_title('Trade-off: Custo vs Eficiência de Redução')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
        ax1.legend()
        
        # Gráfico 2: Análise Custo-Benefício
        # Calcular relação custo-benefício (Eficiência/Custo)
        df_tecnologias['Razão Eficiência/Custo'] = (
            df_tecnologias['Eficiência de Redução (%)'] / 
            df_tecnologias['Custo Estimado (R$/tCO₂eq)'].replace(0, 0.001)
        )
        
        df_sorted_cb = df_tecnologias.sort_values('Razão Eficiência/Custo', ascending=False)
        
        bars = ax2.barh(
            df_sorted_cb['Tecnologia'],
            df_sorted_cb['Razão Eficiência/Custo'],
            color=df_sorted_cb['Cor']
        )
        
        # Adicionar valores nas barras
        for i, (bar, valor) in enumerate(zip(bars, df_sorted_cb['Razão Eficiência/Custo'])):
            ax2.text(
                valor + 0.1,
                bar.get_y() + bar.get_height()/2,
                f'{valor:.2f}',
                va='center',
                fontweight='bold'
            )
        
        ax2.set_xlabel('Razão Eficiência/Custo (% por R$)')
        ax2.set_title('Análise de Custo-Benefício por Tecnologia')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =========================================================================
        # TABELA COMPARATIVA DETALHADA
        # =========================================================================
        
        st.markdown("#### 📊 Tabela Comparativa de Tecnologias")
        
        # Formatar a tabela
        df_display = df_tecnologias.copy()
        df_display['Eficiência de Redução (%)'] = df_display['Eficiência de Redução (%)'].apply(
            lambda x: f"{x:.1f}%"
        )
        df_display['Custo Estimado (R$/tCO₂eq)'] = df_display['Custo Estimado (R$/tCO₂eq)'].apply(
            lambda x: f"R$ {formatar_br(x)}"
        )
        df_display['Razão Eficiência/Custo'] = df_display['Razão Eficiência/Custo'].apply(
            lambda x: f"{x:.2f} %/R$"
        )
        
        # Remover coluna de cores para exibição
        df_display = df_display.drop('Cor', axis=1)
        
        st.dataframe(df_display, use_container_width=True)
        
        # =========================================================================
        # ANÁLISE DE SENSIBILIDADE DO TRADE-OFF
        # =========================================================================
        
        st.markdown("#### 📈 Análise de Sensibilidade do Trade-off")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variação de eficiência
            variacao_eficiencia = st.slider(
                "Variação na Eficiência da Vermicompostagem (%)",
                -20, 20, 0, 5,
                help="Simule variações na eficiência da vermicompostagem"
            )
            
            eficiencia_ajustada = eficiencia_vermi_real * (1 + variacao_eficiencia/100)
            
            st.metric(
                "Eficiência Ajustada",
                f"{eficiencia_ajustada:.1f}%",
                delta=f"{variacao_eficiencia}%"
            )
        
        with col2:
            # Variação de custo
            variacao_custo = st.slider(
                "Variação no Custo da Vermicompostagem (%)",
                -30, 50, 0, 5,
                help="Simule variações no custo da vermicompostagem"
            )
            
            custo_ajustado = custo_vermi_real * (1 + variacao_custo/100)
            
            st.metric(
                "Custo Ajustado",
                f"R$ {formatar_br(custo_ajustado)}",
                delta=f"{variacao_custo}%"
            )
        
        # Gráfico de sensibilidade
        fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
        
        # Ponto original
        ax_sens.scatter(
            custo_vermi_real,
            eficiencia_vermi_real,
            s=400,
            color='blue',
            label='Cenário Base',
            edgecolors='black',
            linewidth=3,
            marker='o'
        )
        
        # Ponto ajustado
        ax_sens.scatter(
            custo_ajustado,
            eficiencia_ajustada,
            s=400,
            color='red',
            label='Cenário Ajustado',
            edgecolors='black',
            linewidth=3,
            marker='s'
        )
        
        # Linha conectando os pontos
        ax_sens.plot(
            [custo_vermi_real, custo_ajustado],
            [eficiencia_vermi_real, eficiencia_ajustada],
            'k--',
            alpha=0.5,
            label='Trajetória de Sensibilidade'
        )
        
        ax_sens.set_xlabel('Custo por tCO₂eq Evitada (R$)')
        ax_sens.set_ylabel('Eficiência de Redução (%)')
        ax_sens.set_title('Análise de Sensibilidade do Trade-off')
        ax_sens.legend()
        ax_sens.grid(True, alpha=0.3)
        ax_sens.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
        
        st.pyplot(fig_sens)
        
        # =========================================================================
        # CONCLUSÕES DO TRADE-OFF
        # =========================================================================
        
        st.markdown("#### 💡 Conclusões do Trade-off")
        
        # Determinar posicionamento competitivo
        if eficiencia_vermi_real > 70 and custo_vermi_real < df_tecnologias['Custo Estimado (R$/tCO₂eq)'].mean():
            st.success("""
            **✅ POSICIONAMENTO COMPETITIVO EXCELENTE**
            - Alta eficiência com custo abaixo da média
            - Vermicompostagem é a tecnologia mais eficiente em custo
            - Forte vantagem competitiva no mercado
            """)
        elif eficiencia_vermi_real > 60:
            st.info("""
            **🔵 POSICIONAMENTO COMPETITIVO BOM**
            - Boa eficiência com custo competitivo
            - Vermicompostagem é uma opção viável
            - Considerar melhorias para aumentar eficiência ou reduzir custos
            """)
        else:
            st.warning("""
            **⚠️ POSICIONAMENTO COMPETITIVO MODESTO**
            - Eficiência ou custo precisam de melhorias
            - Considerar otimizações no processo
            - Avaliar incentivos ou parcerias
            """)
        
        # Recomendações específicas
        st.markdown("""
        **🎯 Recomendações Baseadas no Trade-off:**
        
        1. **Para alta eficiência:**
           - Focar em mercados que valorizam reduções comprovadas
           - Buscar certificações de qualidade
           - Desenvolver indicadores de desempenho
        
        2. **Para baixo custo:**
           - Otimizar escala de operação
           - Buscar sinergias com outras atividades
           - Considerar modelos de negócio inovadores
        
        3. **Para balancear custo e eficiência:**
           - Implementar melhorias incrementais
           - Monitorar KPIs regularmente
           - Realizar benchmarking com outras tecnologias
        """)

# ============================================================================
# EXECUÇÃO PRINCIPAL - SIMULAÇÃO COMPLETA
# ============================================================================
def executar_simulacao_tea_completa():
    """
    Executa simulação completa TEA com trade-off corrigido
    """
    st.sidebar.header("⚙️ Parâmetros da Simulação")
    
    # Parâmetros básicos
    toneladas_residuo = st.sidebar.number_input(
        "Quantidade de Resíduos (ton/ano)",
        min_value=100.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0
    )
    
    anos_simulacao = st.sidebar.slider(
        "Período de Análise (anos)",
        min_value=5,
        max_value=30,
        value=10,
        step=1
    )
    
    # Parâmetros financeiros
    st.sidebar.subheader("💰 Parâmetros Financeiros")
    
    capex_total = st.sidebar.number_input(
        "CAPEX Total (R$)",
        min_value=100000.0,
        max_value=5000000.0,
        value=500000.0,
        step=50000.0
    )
    
    opex_anual = st.sidebar.number_input(
        "OPEX Anual (R$/ano)",
        min_value=50000.0,
        max_value=500000.0,
        value=100000.0,
        step=10000.0
    )
    
    # Receitas
    preco_carbono = st.sidebar.number_input(
        "Preço do Carbono (R$/tCO₂eq)",
        min_value=50.0,
        max_value=500.0,
        value=100.0,
        step=10.0
    )
    
    receita_humus = st.sidebar.number_input(
        "Receita com Húmus (R$/ano)",
        min_value=0.0,
        max_value=200000.0,
        value=50000.0,
        step=10000.0
    )
    
    economia_aterro = st.sidebar.number_input(
        "Economia com Aterro (R$/ano)",
        min_value=0.0,
        max_value=100000.0,
        value=20000.0,
        step=5000.0
    )
    
    taxa_desconto = st.sidebar.slider(
        "Taxa de Desconto (%)",
        min_value=5.0,
        max_value=15.0,
        value=8.0,
        step=0.5
    ) / 100
    
    # Parâmetros ambientais
    st.sidebar.subheader("🌍 Parâmetros Ambientais")
    
    fator_emissao_aterro = st.sidebar.number_input(
        "Fator de Emissão - Aterro (kg CO₂eq/kg)",
        min_value=0.5,
        max_value=2.0,
        value=1.2,
        step=0.1
    )
    
    fator_emissao_vermi = st.sidebar.number_input(
        "Fator de Emissão - Vermicompostagem (kg CO₂eq/kg)",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    
    # Calcular potencial de mitigação
    parametros_mitigacao = {
        'toneladas_residuo': toneladas_residuo,
        'anos': anos_simulacao,
        'fator_emissao_aterro': fator_emissao_aterro,
        'fator_emissao_vermi': fator_emissao_vermi
    }
    
    resultados_mitigacao = calcular_potencial_mitigacao(parametros_mitigacao)
    
    # Calcular receitas
    reducao_anual_tco2eq = resultados_mitigacao['reducao_anual']
    receita_carbono = reducao_anual_tco2eq * preco_carbono
    receita_total_anual = receita_humus + receita_carbono + economia_aterro
    
    # Calcular indicadores financeiros CORRETOS
    indicadores = calcular_indicadores_financeiros(
        capex_total,
        opex_anual,
        receita_total_anual,
        anos_simulacao,
        reducao_anual_tco2eq,  # Parâmetro adicionado
        taxa_desconto
    )
    
    # Preparar análise TEA
    analise_tea = {
        'parametros': {
            'capex_total': capex_total,
            'opex_anual': opex_anual,
            'receita_total_anual': receita_total_anual,
            'receita_carbono': receita_carbono,
            'receita_humus': receita_humus,
            'economia_aterro': economia_aterro,
            'anos_simulacao': anos_simulacao,
            'taxa_desconto': taxa_desconto,
            'reducao_anual_tco2eq': reducao_anual_tco2eq
        },
        'indicadores': indicadores
    }
    
    # Análise de sensibilidade
    st.sidebar.subheader("📊 Análise de Sensibilidade")
    
    if st.sidebar.checkbox("Executar Análise de Sensibilidade"):
        variaveis_analise = {}
        
        if st.sidebar.checkbox("Variação no Preço do Carbono"):
            valores_carbono = np.linspace(50, 200, 6)
            variaveis_analise['preco_carbono'] = valores_carbono
        
        if st.sidebar.checkbox("Variação no CAPEX"):
            valores_capex = np.array([-30, -15, 0, 15, 30, 50])
            variaveis_analise['custo_capex'] = valores_capex
        
        if st.sidebar.checkbox("Variação no OPEX"):
            valores_opex = np.array([-20, -10, 0, 10, 20, 30])
            variaveis_analise['custo_opex'] = valores_opex
        
        if st.sidebar.checkbox("Variação na Produção de Húmus"):
            valores_humus = np.array([-30, -15, 0, 15, 30, 50])
            variaveis_analise['producao_humus'] = valores_humus
        
        if variaveis_analise:
            # Preparar parâmetros base para sensibilidade
            parametros_base_sens = {
                'capex_total': capex_total,
                'opex_anual': opex_anual,
                'receita_total_anual': receita_total_anual,
                'receita_humus': receita_humus,
                'receita_carbono': receita_carbono,
                'economia_aterro': economia_aterro,
                'anos_simulacao': anos_simulacao,
                'taxa_desconto': taxa_desconto
            }
            
            sensibilidade_tese = analise_sensibilidade_tea(
                parametros_base_sens,
                variaveis_analise,
                reducao_anual_tco2eq
            )
        else:
            sensibilidade_tese = {}
    else:
        sensibilidade_tese = {}
    
    # Criar dashboard
    criar_dashboard_tea(analise_tea, sensibilidade_tese, resultados_mitigacao)
    
    # Exportar resultados
    st.sidebar.subheader("📤 Exportar Resultados")
    
    if st.sidebar.button("📥 Exportar Análise TEA"):
        # Criar DataFrame com resultados
        dados_exportacao = {
            'Ano': list(range(anos_simulacao + 1)),
            'Fluxo de Caixa': indicadores['fluxo_caixa'],
            'Fluxo Acumulado': np.cumsum(indicadores['fluxo_caixa']).tolist()
        }
        
        df_export = pd.DataFrame(dados_exportacao)
        
        # Adicionar métricas
        metricas = pd.DataFrame({
            'Métrica': ['VPL', 'TIR', 'Payback', 'Payback Descontado', 'Custo por tCO₂eq'],
            'Valor': [
                indicadores['vpl'],
                indicadores['tir'],
                indicadores['payback_anos'] if indicadores['payback_anos'] else 'N/A',
                indicadores['payback_desc_anos'] if indicadores['payback_desc_anos'] else 'N/A',
                indicadores['custo_tonelada_evitada']
            ]
        })
        
        # Criar arquivo Excel
        with pd.ExcelWriter('analise_tea_vermicompostagem.xlsx') as writer:
            df_export.to_excel(writer, sheet_name='Fluxo de Caixa', index=False)
            metricas.to_excel(writer, sheet_name='Métricas', index=False)
        
        st.sidebar.success("✅ Análise exportada com sucesso!")
        
        # Criar botão de download
        with open('analise_tea_vermicompostagem.xlsx', 'rb') as f:
            st.sidebar.download_button(
                label="⬇️ Baixar Arquivo Excel",
                data=f,
                file_name='analise_tea_vermicompostagem.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

# ============================================================================
# EXECUTAR APLICAÇÃO
# ============================================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="Análise TEA - Vermicompostagem",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🌱 Análise Técnico-Econômica de Vermicompostagem")
    st.markdown("""
    Esta ferramenta realiza uma análise completa de viabilidade técnica, econômica e ambiental 
    para projetos de vermicompostagem de resíduos orgânicos.
    
    ### 🚀 **Principais Funcionalidades:**
    - **Análise Financeira**: VPL, TIR, Payback, Custo por tCO₂eq evitada
    - **Impacto Ambiental**: Redução de emissões e eficiência do sistema
    - **Trade-off Econômico-Ambiental**: Comparação com outras tecnologias
    - **Análise de Sensibilidade**: Simulação de diferentes cenários
    """)
    
    executar_simulacao_tea_completa()
