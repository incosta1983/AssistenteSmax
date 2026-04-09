import streamlit as st
import httpx
from pinecone import Pinecone

# --- CONFIGURAÇÕES SEGURAS DO COFRE ---
OPENAI_KEY = st.secrets["OPENAI_KEY"]
PINECONE_KEY = st.secrets["PINECONE_KEY"]

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("eproc-chamados") 

st.set_page_config(page_title="Eproc AI Support", layout="wide")
st.title("⚖️ Assistente Inteligente Eproc")
st.markdown("Base de dados: **Pinecone** | Cérebro da IA: **OpenAI (GPT-4o-Mini)**")

# --- AMORTECEDORES UNIVERSAIS (Evitam erros de Lista vs Dicionário) ---
def safe_get(obj, key, default=""):
    """Puxa a informação não importa o formato que a IA mandar."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, key):
        return getattr(obj, key, default)
    return default

def cacar_embedding(dados):
    """Encontra o vetor matemático em qualquer formato."""
    if isinstance(dados, dict):
        if "embedding" in dados and isinstance(dados["embedding"], list): return dados["embedding"]
        for valor in dados.values():
            resultado = cacar_embedding(valor)
            if resultado: return resultado
    elif isinstance(dados, list):
        for item in dados:
            resultado = cacar_embedding(item)
            if resultado: return resultado
    return None

# --- INÍCIO DA INTERFACE ---
pergunta = st.text_area("Descreva o problema:", height=150)

if st.button("🔍 Gerar Solução"):
    if not pergunta:
        st.warning("⚠️ Digite algo para buscar.")
    else:
        with st.spinner("Consultando a Base de Conhecimento..."):
            try:
                # 1. Gerar Vetor (OpenAI)
                headers_oa = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
                res_oa = httpx.post("https://api.openai.com/v1/embeddings", json={"input": pergunta, "model": "text-embedding-3-small"}, headers=headers_oa, timeout=20.0)
                
                embedding = cacar_embedding(res_oa.json())
                if not embedding:
                    st.error("Erro ao gerar vetor na OpenAI.")
                    st.stop()

                # 2. Busca no Pinecone
                resultados = index.query(vector=embedding, top_k=3, include_metadata=True)
                
                # Usa o amortecedor para pegar os resultados
                matches = safe_get(resultados, "matches", [])

                if not matches or len(matches) == 0:
                    st.warning("Nenhum chamado similar encontrado.")
                else:
                    contexto = ""
                    # 3. Formata os resultados
                    for r in matches:
                        meta = safe_get(r, "metadata", {})
                        cid = safe_get(meta, "id_chamado_original", "N/A")
                        sol = safe_get(meta, "resposta_limpa", "Sem conteúdo")
                        contexto += f"CHAMADO #{cid}: {sol}\n\n"

                    # 4. OpenAI GPT gera a resposta final
                    prompt_sistema = "Você é o Assistente de Suporte Eproc. Use os casos abaixo para resolver o problema do usuário de forma clara e educada."
                    prompt_usuario = f"Casos Históricos:\n{contexto}\nProblema do Usuário: {pergunta}"
                    
                    payload_chat = {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": prompt_sistema},
                            {"role": "user", "content": prompt_usuario}
                        ]
                    }
                    
                    res_chat = httpx.post("https://api.openai.com/v1/chat/completions", json=payload_chat, headers=headers_oa, timeout=30.0)
                    dados_chat = res_chat.json()
                    
                    # Usa o amortecedor para ler a resposta do GPT
                    choices = safe_get(dados_chat, "choices", [])
                    if isinstance(choices, list) and len(choices) > 0:
                        primeira_opcao = choices
                        mensagem = safe_get(primeira_opcao, "message", {})
                        resposta_final = safe_get(mensagem, "content", "Erro ao ler a resposta.")
                        
                        st.success("✅ Solução sugerida:")
                        st.markdown(resposta_final)
                        
                        st.divider()
                        # Exibe as referências
                        for r in matches:
                            meta = safe_get(r, "metadata", {})
                            score = safe_get(r, "score", 0.0)
                            cid = safe_get(meta, "id_chamado_original", "N/A")
                            with st.expander(f"Ref: Chamado #{cid} (Similaridade: {score:.2f})"):
                                st.write(safe_get(meta, "resposta_limpa", "Sem texto"))
                    else:
                        st.error(f"A OpenAI retornou um formato inesperado. Detalhes: {dados_chat}")

            except Exception as e:
                st.error(f"Erro Crítico: {e}")
