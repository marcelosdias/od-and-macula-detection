from ultralytics import YOLO

# Carrega o modelo YOLO pré-treinado
model = YOLO('yolo11x.pt')

# Treina o modelo
train_results = model.train(
    data='coco.yaml',   # Especifica o arquivo de configuração do dataset
    val=False,          # Desativa a validação durante o treino
    epochs=1000,        # Número de épocas e tamanho do lote
    batch=32, 

    imgsz=320,          # Tamanho das imagens de entrada

    patience=100,       # Número de épocas para esperar antes de interromper o treino por falta de melhora
    close_mosaic=50     # Número de épocas antes de desativar mosaico

    dropout=0.1,        # Probabilidade de dropout para regularização

    # Configurações de augmentação de dados
    degrees= 15.0,      # Rotação máxima em graus
    translate= 0.1,     # Translação máxima como fração da dimensão da imagem
    scale= 0.5,         # Escalamento máximo
    fliplr= 0.5,        # Probabilidade de flip horizontal
    flipud=0.5,         # Probabilidade de flip vertical
    mosaic= 1.0,        # Probabilidade de aplicar mosaico
    mixup= 0.2,         # Probabilidade de aplicar mixup
    
    # Hiperparâmetros de perda
    cls= 0.7,           # Peso para perda de classificação
    box= 7.5,           # Peso para perda de localização de caixas

)
