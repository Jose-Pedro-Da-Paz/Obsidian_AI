import streamlit as st
from groq import Groq
from transformers import AutoModel, AutoTokenizer
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# Inicializar o modelo e o tokenizer
model_name = 'jinaai/jina-embeddings-v2-base-en'
embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Caminho para os embeddings
embeddings_file = "embeddings.pkl"
markdown_folder = '' #your obsidian archives

# Função para gerar embeddings
def generate_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embedding = embedding_model(**tokens).last_hidden_state.mean(dim=1)
    return embedding.numpy().flatten()

# Função para ler arquivos Markdown
def read_markdown_files(folder_path):
    markdown_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                markdown_data.append({'file': file, 'path': file_path, 'content': content})
    return markdown_data


# Função para realizar a consulta e retornar os arquivos mais relevantes
def query_markdown(query, markdown_data, top_k=5):
    query_embedding = generate_embedding(query)
    similarities = []

    for item in markdown_data:
        if isinstance(item, dict):  # Verificar se o item é um dicionário
            if 'embedding' in item and isinstance(item['embedding'], np.ndarray):
                sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                similarities.append({'file': item.get('file', 'desconhecido'), 'path': item['path'], 'similarity': sim})
            else:
                st.warning(f"O arquivo {item.get('file', 'desconhecido')} não possui embedding válido.")
        else:
            st.warning(f"Item inválido encontrado: {item}")

    # Retornar os itens mais similares
    return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:top_k]

# Função para preparar o contexto
def prepare_context(results, markdown_data):
    matched_texts = []
    for result in results:
        for item in markdown_data:
            if item['path'] == result['path']:
                matched_texts.append(item['content'])
    context = "\n\n".join(matched_texts)
    return context

# Função para gerar o prompt completo
def generate_prompt(context, query):
    sys_prompt = f"""
    Instruções:
    - O seu nome é Obsidian, e você é a LLM responsável por responder as perguntas relacionadas as anotações pessoais
    que os usuários criam no programa chamado Obsidian.
    - Seja atencioso e responda as questões que forem perguntadas.
    - Utilize o contexto que for encaminhado para responder as perguntas.
    - Se não souber responder uma questão responda "Infelizmente, não consigo lhe responder isso".
    
    Context: {context}

    User Query: {query}
    """
    return sys_prompt

# Função para enviar a consulta para a API GROQ
def query_groq(prompt):
    client = Groq(api_key="YOUR API KEY FROM GROQ")
    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # Modelo escolhido
        messages=[{"role": "system", "content": prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True
    )
    
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

# Função para carregar ou gerar embeddings
def load_or_generate_embeddings():
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    else:
        markdown_data = read_markdown_files(markdown_folder)
        for item in markdown_data:
            try:
                item['embedding'] = generate_embedding(item['content'])
            except Exception as e:
                st.error(f"Erro ao gerar embedding para o arquivo {item.get('file', 'desconhecido')}: {e}")
        with open(embeddings_file, 'wb') as f:
            pickle.dump(markdown_data, f)
        return markdown_data


# Interface com Streamlit
st.set_page_config(page_title="Obsidian Chat", layout="wide")
st.title("Obsidian Chat - Suporte às suas Notas")

# Botão para atualizar embeddings
if st.button("Atualizar Embeddings"):
    markdown_data = read_markdown_files(markdown_folder)
    for item in markdown_data:
        item['embedding'] = generate_embedding(item['content'])
    with open(embeddings_file, 'wb') as f:
        pickle.dump(markdown_data, f)
    st.success("Embeddings atualizados com sucesso!")

# Carregar embeddings
st.sidebar.header("Carregar Embeddings")
if "markdown_data" not in st.session_state:
    st.session_state["markdown_data"] = load_or_generate_embeddings()

# Histórico do Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico do Chat
st.sidebar.header("Histórico do Chat")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**Você:** {msg['content']}")
    else:
        st.markdown(f"**Obsidian:** {msg['content']}")

# Campo de entrada do usuário
user_input = st.text_input("Digite sua pergunta:")

if user_input:
    # Realizar a consulta
    results = query_markdown(user_input, st.session_state["markdown_data"])
    context = prepare_context(results, st.session_state["markdown_data"])
    prompt = generate_prompt(context, user_input)
    response = query_groq(prompt)

    # Atualizar histórico
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Exibir resposta
    st.markdown(f"**Obsidian:** {response}")
