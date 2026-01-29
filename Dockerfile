# Imagem leve do python
FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia os requisitos primeiro
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto para dentro do container
COPY . .

# Expõe a porta que o FastAPI usa
EXPOSE 8000

# Comando para rodar a aplicação
# --host 0.0.0.0 é obrigatório no Docker para ser acessível fora
CMD ["uvicorn", "web.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]