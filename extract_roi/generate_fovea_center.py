import os
import json
import cv2
from tqdm import tqdm

# Função para converter coordenadas normalizadas para coordenadas originais
def yolo_to_original_coordinates(x_center, y_center, width, height, img_width, img_height):
    x = x_center #* img_width
    y = y_center #* img_height
    return [x, y] 

for type in ['test']:
  input_folder = f'../modelo/runs/detect/predict/labels'
  images_dir = f'../dataset/processed/{type}/images'
  output_file = f'../dataset/fovea-center/{type}-fovea-center.json'

  predictions = {}

  print(f"\nProcessando - {type}")

  # Itera sobre todos os arquivos de predição
  pred_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        
  # Adiciona uma barra de progresso
  for pred_file in tqdm(pred_files, desc='Processando arquivos', unit='arquivo'):
            img_file = pred_file.replace('.txt', '.jpg')
            img_path = os.path.join(images_dir, img_file)
            
            # Verifica se a imagem correspondente existe
            if os.path.exists(img_path):
                # Usa OpenCV para obter as dimensões da imagem
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erro ao ler a imagem {img_path}")
                    continue
                img_height, img_width = img.shape[:2]
                
                with open(os.path.join(input_folder, pred_file), 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    if int(class_id) == 0:
                        coordinates = yolo_to_original_coordinates(x_center, y_center, width, height, img_width, img_height)
                        if img_file not in predictions:
                            predictions[img_file] = []
                        predictions[img_file].append(coordinates)
               
  # Salva as predições no formato JSON
  with open(output_file, 'w') as json_file:
    json.dump(predictions, json_file, indent=4)
  print(f'Predições salvas em {output_file}')