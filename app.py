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

# ── Funções Auxiliares com Cache ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo BERT... (Pode levar alguns minutos na primeira execução)")
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

# Inicializa o modelo IA
sentiment_analyzer = carregar_modelo()

# ── Barra Lateral (Sidebar) ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Sentimento 2025")
    st.markdown("""
    Este aplicativo realiza análise de sentimentos em português utilizando Inteligência Artificial.
    
    **Escala de Classificação:**
    """)
    
    dados_legenda = {"Estrelas": ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], 
                     "Significado": list(MAPA_ESTRELAS.values())}
    st.table(pd.DataFrame(dados_legenda))

# ── Corpo Principal da Aplicação ──────────────────────────────────────────────
st.title("🧠 Análise de Sentimentos com Transformers")
st.markdown("Classifique textos e feedbacks em lote ou individualmente.")

# Criação das Abas
aba_individual, aba_lote = st.tabs(["📝 Análise Individual", "📂 Análise em Lote (Arquivo)"])

# ── ABA 1: Análise Individual ────────────────────────────────────────────────
with aba_individual:
    st.header("Teste Rápido de Frases")
    frase_usuario = st.text_area("Sua frase/opinião:", placeholder="Escreva aqui...")
    
    if st.button("Analisar Frase", type="primary"):
        if frase_usuario.strip():
            with st.spinner("Analisando..."):
                try:
                    frase_limpa = limpar_frase(frase_usuario)
                    resultado = sentiment_analyzer(frase_limpa)[0]
                    
                    estrelas = extrair_estrelas(resultado["label"])
                    confianca = resultado["score"]
                    categoria = MAPA_ESTRELAS[estrelas]
                    
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Classificação", value="⭐" * estrelas)
                    col2.metric(label="Sentimento", value=categoria.split()[0] + " " + categoria.split()[1])
                    col3.metric(label="Confiança do Modelo", value=f"{confianca:.2%}")
                        
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")
        else:
            st.warning("Por favor, digite algum texto.")

# ── ABA 2: Análise em Lote ───────────────────────────────────────────────────
with aba_lote:
    st.header("Análise de Arquivos")
    st.markdown("Faça o upload de um arquivo de texto (**`.txt`**) com uma opinião por linha.")
    
    arquivo_carregado = st.file_uploader("Escolha um arquivo .txt", type=["txt"])
    
    if arquivo_carregado is not None:
        conteudo = arquivo_carregado.read().decode("utf-8")
        linhas = conteudo.splitlines()
        frases_arquivo = [limpar_frase(linha) for line in linhas if (linha := line.strip())]
        
        st.info(f"📋 O arquivo contém **{len(frases_arquivo)}** frase(s) prontas para processamento.")
        
        if st.button("Iniciar Processamento em Lote", type="primary"):
            resultados_lote = []
            contagem_estrelas = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            barra_progresso = st.progress(0)
            status_texto = st.empty()
            
            for i, frase in enumerate(frases_arquivo):
                status_texto.text(f"Processando {i+1} de {len(frases_arquivo)}...")
                try:
                    resultado = sentiment_analyzer(frase)[0]
                    estrelas = extrair_estrelas(resultado["label"])
                    confianca = resultado["score"]
                    
                    contagem_estrelas[estrelas] += 1
                    resultados_lote.append({
                        "Frase": frase,
                        "Classificação": "⭐" * estrelas,
                        "Sentimento": MAPA_ESTRELAS[estrelas],
                        "Confiança": f"{confianca:.2%}"
                    })
                except Exception as e:
                    resultados_lote.append({
                        "Frase": frase, "Classificação": "⚠️ Erro", "Sentimento": f"Erro: {e}", "Confiança": "0%"
                    })
                
                barra_progresso.progress((i + 1) / len(frases_arquivo))
                
            status_texto.text("✅ Processamento concluído!")
            
            # DataFrame dos resultados
            df_resultados = pd.DataFrame(resultados_lote)
            st.markdown("### 📊 Dados Processados")
            st.dataframe(df_resultados, use_container_width=True)
            
            # Download CSV
            csv_data = df_resultados.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(label="📥 Baixar Resultados em CSV", data=csv_data, file_name="resultados_sentimento.csv", mime="text/csv")
            
            # ── GRÁFICO NATIVO DO STREAMLIT (Sem Matplotlib!) ─────────────────
            st.markdown("---")
            st.markdown("### 📈 Distribuição de Sentimentos")
            
            # Criando tabela para o gráfico de barras
            df_grafico = pd.DataFrame({
                "Quantidade de Frases": [contagem_estrelas[e] for e in range(1, 6)]
            }, index=["1⭐ Muito Negativo", "2⭐ Negativo", "3⭐ Neutro", "4⭐ Positivo", "5⭐ Muito Positivo"])
            
            # Renderiza o gráfico de barras interativo
            st.bar_chart(df_grafico)
