import glob
import json
import os
import random
from itertools import chain

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from util import sanitize_pixels

# Define a semente para reprodutibilidade
random.seed(42)

# Definição de caminhos
MAIN_PATH = '../../dataset/datasets-cropping-tilling'
DATASET = '../../dataset/{DATASET_NAME}'
DATASET_OUT = '../../dataset/datasets-cropping-tilling/{DATASET_NAME}/'
FOLDER_PATH = '{DATASET}/{DATASET_TYPE}/*.json'

# Inicializa barra de progresso
pbar = tqdm()

# Função para aplicar recorte em imagens e anotações
def albumentations_crop(image, annotations, x_min, y_min, x_max, y_max):
    # Define transformações para recorte
    transforms = [A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True)]
    transform = A.Compose(transforms=transforms, bbox_params=A.BboxParams(format='coco'))
    
    # Aplica transformação
    transformed = transform(image=image, bboxes=annotations)
    return transformed['image'], transformed['bboxes']

# Função para recortar imagem com base na detecção de círculos (Hough Transform)
def hough_crop_coco(image, annotations):
    # Converte a imagem para escala de cinza e aplica desfoque
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 31)

    # Detecta círculos na imagem
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=70, param2=50)
    if circles is None:
        # Se não houver círculos detectados, retorna imagem original
        return image, annotations

    # Extrai as coordenadas do círculo detectado
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    # Define limites do recorte
    height, width, _ = image.shape
    x_min, y_min = int(x - r), int(y - r)
    x_max, y_max = int(x_min + 2 * r), int(y_min + 2 * r)

    # Ajusta os limites do recorte para estar dentro dos limites da imagem
    x_min, y_min, x_max, y_max = sanitize_pixels(
        x_min=x_min, y_min=y_min,
        x_max=x_max, y_max=y_max,
        width=width, height=height,
    )

    # Realiza o recorte usando a função definida
    cropped_image, cropped_bboxes = albumentations_crop(image, annotations, x_min, y_min, x_max, y_max)

    # Ajusta as anotações após o recorte
    new_height, new_width, _ = cropped_image.shape
    scale_x = new_width / (x_max - x_min)
    scale_y = new_height / (y_max - y_min)

    adjusted_bboxes = []
    for bbox in cropped_bboxes:
        x, y, w, h, category_id = bbox
        adjusted_bbox = [
            x * scale_x,
            y * scale_y,
            w * scale_x,
            h * scale_y,
            category_id
        ]
        adjusted_bboxes.append(adjusted_bbox)

    return cropped_image, adjusted_bboxes

# Função para processar imagens e anotações
def process(image, dataset_type, anno, dataset, dataset_out):
    # Caminho de entrada e saída da imagem
    input_image_path = f'{dataset}/{dataset_type}/images/{image["file_name"]}'
    output_image_path = f'{dataset_out}/{dataset_type}/{image["file_name"]}'

    # Lê imagem e prepara anotações
    cv2_image = cv2.imread(filename=input_image_path)
    img_annotations = [
        [x['bbox'][0], x['bbox'][1], x['bbox'][2], x['bbox'][3], x['category_id']]
        for x in anno['annotations'] if x['image_id'] == image['id']
    ]

    # Realiza o recorte e salva a imagem
    cv2_image, img_annotations = hough_crop_coco(cv2_image, img_annotations)
    cv2.imwrite(filename=output_image_path, img=cv2_image)

    # Atualiza barra de progresso
    if pbar.desc != dataset_type:
        pbar.reset()
        pbar.total = len(anno['images'])
        pbar.desc = dataset_type

    pbar.update(1)

    # Retorna novas anotações ajustadas
    return [{
        'id': index + 1,
        'image_id': image['id'],
        'bbox': bbox[:4],
        'category_id': bbox[4],
        'area': bbox[2] * bbox[3],
        'iscrowd': 0
    } for index, bbox in enumerate(img_annotations)]

# Função principal
def main():
    # Cria os diretórios necessários
    os.makedirs(MAIN_PATH, exist_ok=True)

    for selected_dataset in ['idrid']:
        out_path = DATASET_OUT.format(DATASET_NAME=selected_dataset)
        os.makedirs(out_path, exist_ok=True)

        for dataset_type in ['train', 'test']:
            os.makedirs(f'{out_path}/{dataset_type}', exist_ok=True)
            path = DATASET.format(DATASET_NAME=selected_dataset)

            # Processa cada arquivo de anotações
            for anno_path in glob.iglob(FOLDER_PATH.format(DATASET=path, DATASET_TYPE=dataset_type)):
                with open(anno_path, 'r') as f_in:
                    anno = json.load(f_in)
                    new_annos = []
                    for image in anno['images']:
                        new_annos.extend(process(image, dataset_type, anno, path, out_path))

                    # Salva novas anotações
                    with open(anno_path.replace(path, out_path), 'w') as f_out:
                        anno['annotations'] = new_annos
                        json.dump(anno, f_out)

# Executa o programa
if __name__ == '__main__':
    main()
