from ultralytics import YOLO

# Caminho para os pesos do modelo treinado
model_path = "CAMINHO_ATE_PESOS" # Substituir pelo caminho real dos pesos

# Carrega o modelo YOLO treinado
model = YOLO(model_path)

# Caminho para a imagem ou diretório de imagens
source = "CAMINHO_ATE_IMAGEM" # Substituir pelo caminho real das imagens

# Realiza a predição usando o modelo carregado
model.predict(
    source,           # Fonte dos dados (imagem ou diretório)
    task='detect',    # Define a tarefa como detecção de objetos
    save=True,        # Salva as imagens com os resultados
    line_width=8,     # Largura das linhas nas caixas delimitadoras
    save_txt=True     # Salva os resultados em formato de texto
)
