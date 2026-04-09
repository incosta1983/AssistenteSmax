import streamlit as st
import httpx
import google.generativeai as genai

# --- CONFIGURAÇÕES ---
# --- CONFIGURAÇÕES SEGURAS ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_KEY = st.secrets["OPENAI_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]

# Configura o Gemini
genai.configure(api_key=GEMINI_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Eproc AI Support", layout="wide")
st.title("⚖️ Assistente Inteligente Eproc")
st.markdown(f"Base de dados: **72.202 chamados**")

# --- A ARMA SECRETA: FUNÇÃO QUE CAÇA O VETOR EM QUALQUER LUGAR ---
def cacar_embedding(dados):
    # Se for um dicionário, procura a chave "embedding"
    if isinstance(dados, dict):
        if "embedding" in dados and isinstance(dados["embedding"], list):
            return dados["embedding"]
        # Se não achou, procura dentro de todos os valores do dicionário
        for valor in dados.values():
            resultado = cacar_embedding(valor)
            if resultado: return resultado
            
    # Se for uma lista, procura dentro de cada item da lista
    elif isinstance(dados, list):
        for item in dados:
            resultado = cacar_embedding(item)
            if resultado: return resultado
            
    # Se não for nada disso, retorna vazio
    return None

pergunta = st.text_area("Descreva o problema:", height=150)

if st.button("🔍 Gerar Solução"):
    if not pergunta:
        st.warning("⚠️ Digite algo para buscar.")
    else:
        with st.spinner("Consultando os 72 mil chamados..."):
            passo = "Iniciando"
            try:
                passo = "OpenAI"
                headers_oa = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
                data_oa = {"input": pergunta, "model": "text-embedding-3-small"}
                res_oa = httpx.post("https://api.openai.com/v1/embeddings", json=data_oa, headers=headers_oa, timeout=30.0)
                
                passo = "Caçando o Vetor"
                dados_json = res_oa.json()
                
                # Usa a função "Cão Farejador" para achar o vetor não importa onde ele esteja
                embedding = cacar_embedding(dados_json)
                
                if not embedding:
                    st.error("O vetor realmente não está no JSON retornado pela OpenAI.")
                    st.json(dados_json)
                    st.stop()

                passo = "Supabase"
                headers_sb = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
                payload_sb = {"query_embedding": embedding, "match_threshold": 0.3, "match_count": 3}
                res_db = httpx.post(f"{SUPABASE_URL}/rest/v1/rpc/match_chamados", json=payload_sb, headers=headers_sb, timeout=30.0)
                
                resultados = res_db.json()
                
                passo = "Lendo Resultados do Banco"
                # Tratamento para caso o Supabase retorne um erro formal
                if isinstance(resultados, dict) and "message" in resultados:
                    st.error(f"O Supabase negou a busca: {resultados.get('message')}")
                    st.stop()

                # Desembrulha listas extras do Supabase se existirem
                if isinstance(resultados, list) and len(resultados) > 0 and isinstance(resultados, list):
                    resultados = resultados

                if not resultados or not isinstance(resultados, list) or len(resultados) == 0:
                    st.warning("Nenhum chamado similar encontrado nos registros.")
                else:
                    contexto = ""
                    for r in resultados:
                        if isinstance(r, dict):
                            cid = r.get("id_chamado_original", "N/A")
                            sol = r.get("resposta_limpa", "Sem conteúdo")
                            contexto += f"CHAMADO #{cid}: {sol}\n\n"

                    passo = "Gemini"
                    prompt = f"Suporte Eproc. Use os casos abaixo para resolver o problema do usuário:\n{contexto}\nProblema: {pergunta}"
                    resposta_final = model_gemini.generate_content(prompt)
                    
                    st.success("✅ Solução sugerida:")
                    st.markdown(resposta_final.text)
                    
                    st.divider()
                    for r in resultados:
                        if isinstance(r, dict):
                            with st.expander(f"Ref: Chamado #{r.get('id_chamado_original')}"):
                                st.write(r.get('resposta_limpa'))

            except Exception as e:
                st.error(f"🚨 Travou no passo: **{passo}**")
                st.error(f"Erro: {e}")
