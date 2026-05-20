# -*- coding: utf-8 -*-
import subprocess
import sys

# ── AUTO-INSTALADOR DE DEPENDÊNCIAS ──────────────────────────────────────────
# Este bloco detecta se a IA está instalada no servidor. Se não estiver,
# ele força a instalação automaticamente direto pelo código.
try:
    from transformers import pipeline
    import torch
except ModuleNotFoundError:
    # Executa o pip install em segundo plano no servidor do Streamlit
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
    from transformers import pipeline

import streamlit as st
import pandas as pd
import re

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
                    col2.metric(label="Sentimento", value=categoria.split()[0] + " "
