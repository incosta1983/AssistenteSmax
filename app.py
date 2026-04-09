import streamlit as st
import httpx
import google.generativeai as genai
from pinecone import Pinecone

# --- CONFIGURAÇÕES SEGURAS DO COFRE ---
OPENAI_KEY = st.secrets["OPENAI_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]
PINECONE_KEY = st.secrets["PINECONE_KEY"]

# Configurações das IAs e Banco
genai.configure(api_key=GEMINI_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("eproc-chamados") # O nome exato que criamos lá no painel

st.set_page_config(page_title="Eproc AI Support", layout="wide")
st.title("⚖️ Assistente Inteligente Eproc")
st.markdown("Base de dados: **Pinecone Vector Database**")

# A mesma função "Cão Farejador" para achar o vetor sem erro
def cacar_embedding(dados):
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

pergunta = st.text_area("Descreva o problema:", height=150)

if st.button("🔍 Gerar Solução"):
    if not pergunta:
        st.warning("⚠️ Digite algo para buscar.")
    else:
        with st.spinner("Consultando o Pinecone (Ultra Rápido)..."):
            try:
                # 1. Gerar Vetor (OpenAI)
                headers_oa = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
                res_oa = httpx.post("https://api.openai.com/v1/embeddings", json={"input": pergunta, "model": "text-embedding-3-small"}, headers=headers_oa, timeout=20.0)
                embedding = cacar_embedding(res_oa.json())
                
                if not embedding:
                    st.error("Erro ao gerar vetor na OpenAI.")
                    st.stop()

                # 2. Busca no Pinecone (Substitui o Supabase)
                # Pede os 3 mais próximos e inclui os metadados (os textos salvos)
                resultados = index.query(vector=embedding, top_k=3, include_metadata=True)

                if not resultados or "matches" not in resultados or len(resultados["matches"]) == 0:
                    st.warning("Nenhum chamado similar encontrado. (O Pinecone está vazio?)")
                else:
                    contexto = ""
                    # 3. Formata os resultados
                    for r in resultados["matches"]:
                        metadados = r.get("metadata", {})
                        cid = metadados.get("id_chamado_original", "N/A")
                        sol = metadados.get("resposta_limpa", "Sem conteúdo")
                        contexto += f"CHAMADO #{cid}: {sol}\n\n"

                    # 4. Gemini gera a resposta
                    prompt = f"Suporte Eproc. Use os casos abaixo para resolver o problema do usuário:\n{contexto}\nProblema: {pergunta}"
                    resposta_final = model_gemini.generate_content(prompt)
                    
                    st.success("✅ Solução sugerida:")
                    st.markdown(resposta_final.text)
                    
                    st.divider()
                    for r in resultados["matches"]:
                        meta = r.get("metadata", {})
                        with st.expander(f"Ref: Chamado #{meta.get('id_chamado_original', 'N/A')} (Score: {r.get('score', 0):.2f})"):
                            st.write(meta.get('resposta_limpa'))

            except Exception as e:
                st.error(f"Erro Crítico: {e}")
