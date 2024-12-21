import os

import cv2
import shutil

# Define os caminhos para o dataset e a saída
DATASET = '../../dataset/datasets-cropping-tilling/idrid/{dataset_type}'
OUT = "../../dataset/processed/{dataset_type}"

# Cria o diretório de saída, se não existir
if not os.path.exists('../../dataset/processed'):
   os.mkdir('../../dataset/processed')

# Itera sobre os tipos de dados: treino e teste
for type in ['train', 'test']:
   # Lista os arquivos no diretório correspondente
   files = os.listdir(DATASET.format(dataset_type=type))

   for file in files:
      # Define o caminho de saída para o tipo de dado atual
      output = OUT.format(dataset_type=type)

      # Cria o diretório de saída, se não existir
      if not os.path.exists(output):
         os.makedirs(output)

      # Copia os arquivos JSON diretamente para o diretório de saída
      if file.endswith('.json'):
         shutil.copy(DATASET.format(dataset_type=type) + f'/{file}', output)
         continue
      
      # Lê a imagem
      file_path = DATASET.format(dataset_type=type) + f'/{file}'
      image = cv2.imread(file_path)

      # Extrai o canal verde da imagem (geralmente mais informativo em imagens médicas)
      green_channel = image[:,:, 1]

      # Aplica filtro de mediana para reduzir ruído
      image_median_filter = cv2.medianBlur(src=green_channel, ksize=5)

      # Aplica o CLAHE (Contrast Limited Adaptive Histogram Equalization) para melhorar o contraste
      clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
      enhanced_green_channel = clahe.apply(image_median_filter)

      # Aplica filtro bilateral para suavizar mantendo as bordas
      image_bilateral_filter = cv2.bilateralFilter(src=enhanced_green_channel, d=9, sigmaColor=75, sigmaSpace=75)
      
      # Salva a imagem processada no diretório de saída
      cv2.imwrite(OUT.format(dataset_type=type) + f'/images/{file}', enhanced_green_channel)
   
   # Exibe mensagem indicando o fim do processamento
   print(f'Processamendo do {type} finalizado')
