# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# ── Configuração da Página do Streamlit ───────────────────────────────────────
st.set_page_config(
    page_title="Sentimento 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Mapeamento e Constantes ──────────────────────────────────────────────────
NOME_MODELO = "nlptown/bert-base-multilingual-uncased-sentiment"

MAPA_ESTRELAS = {
    1: "Muito Negativo 😠",
    2: "Negativo 😞",
    3: "Neutro / Misto 😐",
    4: "Positivo 🙂",
    5: "Muito Positivo 😄",
}

CORES = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2"]

# ── Funções Auxiliares com Cache ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo BERT do Hugging Face... (Pode levar alguns minutos na primeira execução)")
def carregar_modelo():
    """Carrega o pipeline de análise de sentimento e mantém em cache."""
    return pipeline(
        "sentiment-analysis",
        model=NOME_MODELO,
        truncation=True,
        max_length=512
    )

def extrair_estrelas(label: str) -> int:
    """Extrai o número de estrelas do rótulo (ex: '4 stars' -> 4)."""
    match = re.search(r"(\d)", label)
    return int(match.group(1)) if match else 3

def limpar_frase(texto: str) -> str:
    """Limpa espaços extras e pontos finais da frase."""
    return texto.strip().rstrip('.')

# Inicializa o modelo
sentiment_analyzer = carregar_modelo()

# ── Barra Lateral (Sidebar) ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Sentimento 2025")
    st.markdown("""
    Este aplicativo realiza análise de sentimentos em português utilizando o modelo **BERT Multilíngue** da *NLP Town*.
    
    A classificação usa uma escala de **1 a 5 estrelas**:
    """)
    
    # Tabela de legenda na barra lateral
    dados_legenda = {"Estrelas": ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], 
                     "Significado": list(MAPA_ESTRELAS.values())}
    st.table(pd.DataFrame(dados_legenda))
    
    st.markdown("---")
    st.caption("Desenvolvido para análise automatizada de feedbacks e opiniões.")

# ── Corpo Principal da Aplicação ──────────────────────────────────────────────
st.title("🧠 Análise de Sentimentos com Transformers")
st.markdown("Classifique textos e feedbacks em lote ou individualmente com Inteligência Artificial.")

# Criação das Abas
aba_individual, aba_lote = st.tabs(["📝 Análise Individual", "📂 Análise em Lote (Arquivo)"])

# ── ABA 1: Análise Individual ────────────────────────────────────────────────
with aba_individual:
    st.header("Teste Rápido de Frases")
    st.markdown("Digite uma frase abaixo para verificar o sentimento detectado pelo modelo.")
    
    frase_usuario = st.text_area(
        "Sua frase/opinião:", 
        placeholder="Escreva aqui o que achou do produto ou serviço..."
    )
    
    if st.button("Analisar Frase", type="primary"):
        if frase_usuario.strip():
            with st.spinner("Analisando..."):
                try:
                    frase_limpa = limpar_frase(frase_usuario)
                    resultado = sentiment_analyzer(frase_limpa)[0]
                    
                    estrelas = extrair_estrelas(resultado["label"])
                    confianca = resultado["score"]
                    categoria = MAPA_ESTRELAS[estrelas]
                    
                    # Exibição dos resultados em cards (métricas)
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Classificação", value="⭐" * estrelas)
                    with col2:
                        st.metric(label="Sentimento", value=categoria.split()[0] + " " + categoria.split()[1])
                    with col3:
                        st.metric(label="Confiança do Modelo", value=f"{confianca:.2%}")
                        
                except Exception as e:
                    st.error(f"Erro ao processar a frase: {e}")
        else:
            st.warning("Por favor, digite algum texto antes de analisar.")

# ── ABA 2: Análise em Lote ───────────────────────────────────────────────────
with aba_lote:
    st.header("Análise de Arquivos")
    st.markdown("""
    Faça o upload de um arquivo de texto (**`.txt`**) contendo uma opinião por linha.
    O sistema processará todas as linhas e gerará métricas visuais automaticamente.
    """)
    
    arquivo_carregado = st.file_uploader("Escolha um arquivo .txt", type=["txt"])
    
    if arquivo_carregado is not None:
        # Lendo o arquivo carregado
        conteudo = arquivo_carregado.read().decode("utf-8")
        linhas = conteudo.splitlines()
        frases_arquivo = [limpar_frase(linha) for line in linhas if (linha := line.strip())]
        
        st.info(f"📋 O arquivo contém **{len(frases_arquivo)}** frase(s) válidas prontas para processamento.")
        
        if st.button("Iniciar Processamento em Lote", type="primary"):
            resultados_lote = []
            contagem_estrelas = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            # Barra de progresso do Streamlit
            barra_progresso = st.progress(0)
            status_texto = st.empty()
            
            # Loop de inferência
            for i, frase in enumerate(frases_arquivo):
                status_texto.text(f"Processando frase {i+1} de {len(frases_arquivo)}...")
                try:
                    resultado = sentiment_analyzer(frase)[0]
                    estrelas = extrair_estrelas(resultado["label"])
                    confianca = resultado["score"]
                    
                    contagem_estrelas[estrelas] += 1
                    resultados_lote.append({
                        "Frase": frase,
                        "Estrelas": estrelas,
                        "Classificação": "⭐" * estrelas,
                        "Sentimento": MAPA_ESTRELAS[estrelas],
                        "Confiança": f"{confianca:.2%}",
                        "score_num": confianca
                    })
                except Exception as e:
                    resultados_lote.append({
                        "Frase": frase, "Estrelas": None, "Classificação": "⚠️ Erro", 
                        "Sentimento": f"Erro: {e}", "Confiança": "0.0%", "score_num": 0
                    })
                
                # Atualiza barra de progresso
                barra_progresso.progress((i + 1) / len(frases_arquivo))
                
            status_texto.text("✅ Processamento concluído!")
            
            # Criando DataFrame dos resultados
            df_resultados = pd.DataFrame(resultados_lote)
            
            # ── Exibição da Tabela de Dados ───────────────────────────────────
            st.markdown("### 📊 Dados Processados")
            st.dataframe(df_resultados[["Frase", "Classificação", "Sentimento", "Confiança"]], use_container_width=True)
            
            # Botão de Download para CSV
            csv_data = df_resultados[["Frase", "Estrelas", "Sentimento", "score_num"]].to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Baixar Resultados em CSV (Excel)",
                data=csv_data,
                file_name="resultados_analise_sentimento.csv",
                mime="text/csv"
            )
            
            # ── Geração de Gráficos (Matplotlib) ──────────────────────────────
            st.markdown("---")
            st.markdown("### 📈 Visualização Gráfica")
            
            estrelas_lista = list(range(1, 6))
            totais_lista = [contagem_estrelas[e] for e in estrelas_lista]
            
            # Configuração da figura Matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Distribuição de Sentimentos", fontsize=14, fontweight="bold")
            
            # Gráfico de Barras
            bars = axes[0].bar(estrelas_lista, totais_lista, color=CORES, edgecolor="white", width=0.6)
            axes[0].set_xlabel("Estrelas")
            axes[0].set_ylabel("Quantidade de Frases")
            axes[0].set_title("Quantidade por Categoria")
            axes[0].set_xticks(estrelas_lista)
            axes[0].set_xticklabels([f"{e}⭐" for e in estrelas_lista])
            
            for bar, valor in zip(bars, totais_lista):
                if valor > 0:
                    axes[0].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(valor),
                        ha="center", va="bottom", fontweight="bold"
                    )
            
            # Gráfico de Pizza (Filtrando zeros)
            dados_pizza = [(e, t) for e, t in zip(estrelas_lista, totais_lista) if t > 0]
            if dados_pizza:
                estrelas_p = [d[0] for d in dados_pizza]
                totais_p = [d[1] for d in dados_pizza]
                cores_p = [CORES[e - 1] for e in estrelas_p]
                rotulos_p = [f"{e}★ ({t})" for e, t in dados_pizza]
                
                axes[1].pie(
                    totais_p,
                    labels=rotulos_p,
                    colors=cores_p,
                    autopct="%1.1f%%",
                    startangle=90,
                    pctdistance=0.75
                )
                axes[1].set_title("Proporção por Categoria")
            else:
                axes[1].text(0.5, 0.5, "Sem dados suficientes para pizza", ha="center", va="center")
            
            plt.tight_layout()
            
            # Renderiza o gráfico do Matplotlib nativamente no Streamlit
            st.pyplot(fig)
