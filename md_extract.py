import os
import torch
import pickle
import hashlib
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

# Carrega o modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

# Função para gerar hash do conteúdo do arquivo
def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Função para gerar embeddings
def generate_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embedding = embedding_model(**tokens).last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()

# Caminho dos arquivos Markdown e do arquivo de armazenamento dos embeddings
markdown_dir = "G:\Organizar - Obsidian\Organizar"
embeddings_file = "embeddings.pkl"

# Carrega os embeddings salvos (se existirem)
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
else:
    embeddings_data = {}

# Verifica arquivos no diretório e atualiza embeddings
updated = False
for root, _, files in os.walk(markdown_dir):
    for file_name in files:
        if file_name.endswith('.md'):
            file_path = os.path.join(root, file_name)
            file_hash = get_file_hash(file_path)
            
            # Verifica se o arquivo é novo ou foi modificado
            if file_name not in embeddings_data or embeddings_data[file_name]['hash'] != file_hash:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                embedding = generate_embedding(file_content)
                embeddings_data[file_name] = {
                    'hash': file_hash,
                    'embedding': embedding,
                    'last_modified': datetime.now().isoformat()
                }
                updated = True

# Salva os embeddings atualizados
if updated:
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print("Embeddings atualizados e salvos.")
else:
    print("Nenhuma atualização necessária.")

# Acessando os embeddings
for file_name, data in embeddings_data.items():
    print(f"Arquivo: {file_name}, Última modificação: {data['last_modified']}")

