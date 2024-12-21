from ultralytics import YOLO

# Caminho para os pesos do modelo treinado
model_path = 'CAMINHO_ATE_PESOS' # Substituir pelo caminho real dos pesos

# Carrega o modelo YOLO treinado
model = YOLO(model_path)

# Avalia o modelo nos dados de teste
model.val(
    data='coco.yaml', # Arquivo de configuração do dataset
    split='test',     # Define a divisão de dados para teste
)
