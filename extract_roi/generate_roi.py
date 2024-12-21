import json
import os
from tqdm import tqdm
from PIL import Image

# Função para recortar a ROI da imagem original e retornar os limites da ROI
def crop_roi(image_path, center_x, center_y, roi_width, roi_height):
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Calcular os limites da ROI
    x1 = int(center_x - roi_width / 2)
    y1 = int(center_y - roi_height / 2)
    x2 = int(center_x + roi_width / 2)
    y2 = int(center_y + roi_height / 2)

    # Garantir que os limites estejam dentro da imagem
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    # Recortar a ROI
    roi = image.crop((x1, y1, x2, y2))
    
    return roi, x1, y1, x2, y2

# Função para ajustar as coordenadas das anotações para a nova ROI
def adjust_annotations(annotations, roi_x1, roi_y1, roi_width, roi_height):
    filtered_annotations = []
    for annotation in annotations:
        x, y, width, height = annotation['bbox']
        
        # Calcula os limites da bounding box na imagem original
        box_x1 = x
        box_y1 = y
        box_x2 = x + width
        box_y2 = y + height

        # Verifica se a bounding box está dentro da ROI (parcial ou totalmente)
        if box_x2 > roi_x1 and box_x1 < roi_x1 + roi_width and box_y2 > roi_y1 and box_y1 < roi_y1 + roi_height:
            # Calcula a nova posição da bounding box relativa à ROI
            new_x1 = max(0, box_x1 - roi_x1)
            new_y1 = max(0, box_y1 - roi_y1)
            new_x2 = min(roi_width, box_x2 - roi_x1)
            new_y2 = min(roi_height, box_y2 - roi_y1)

            # Calcula a nova largura e altura da bounding box
            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1
            
            if new_width > 0 and new_height > 0:
                new_annotation = annotation.copy()
                new_annotation['bbox'] = [new_x1, new_y1, new_width, new_height]

                # Ajustar a máscara de segmentação se existir
                if 'segmentation' in annotation:
                    new_segmentation = []
                    for segment in annotation['segmentation']:
                        new_segment = []
                        for i in range(0, len(segment), 2):
                            seg_x = max(0, min(segment[i] - roi_x1, roi_width))
                            seg_y = max(0, min(segment[i + 1] - roi_y1, roi_height))
                            new_segment.extend([seg_x, seg_y])
                        new_segmentation.append(new_segment)
                    new_annotation['segmentation'] = new_segmentation

                filtered_annotations.append(new_annotation)
    return filtered_annotations

def process_images_with_annotations(input_folder, output_folder, coco_annotation_file, fovea_annotation_file, roi_width=1280, roi_height=1280):
    # Cria a pasta de destino se ela não existir
    os.makedirs(output_folder, exist_ok=True)

    # Carrega as anotações COCO e fovea
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
        
    with open(fovea_annotation_file, 'r') as f:
        fovea_data = json.load(f)

    # Novo dicionário para as anotações ajustadas
    new_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco_data["categories"]
    }

    # Adiciona uma barra de progresso
    for image_name, fovea_coords in tqdm(fovea_data.items(), desc='Processando imagens', unit='imagem'):
        center_x, center_y = fovea_coords[0]

        image_name = image_name.split('.')[0] + '.jpg'
        image_path = os.path.join(input_folder, image_name)
        
        # Converter as coordenadas da fovea para a resolução original da imagem
        roi, roi_x1, roi_y1, roi_x2, roi_y2 = crop_roi(image_path, center_x * 1811, center_y * 1811, roi_width, roi_height)
            
        # Salvar a ROI recortada com alta qualidade
        roi_output_path = os.path.join(output_folder, image_name)
        
        # Para salvar a imagem PNG sem compressão (melhor qualidade e maior tamanho de arquivo)
        roi.save(roi_output_path, format='PNG', compress_level=0)

        # Ajustar as anotações para a nova resolução da ROI
        image_id = next(image['id'] for image in coco_data['images'] if image['file_name'] == image_name)
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        new_annotations['images'].append({
            "id": image_id,
            "file_name": image_name,
            "width": roi_x2 - roi_x1,
            "height": roi_y2 - roi_y1
        })

        filtered_annotations = adjust_annotations(annotations, roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)
        new_annotations['annotations'].extend(filtered_annotations)

    # Salvar o novo arquivo de anotações COCO
    output_annotations_file = os.path.join(output_folder, '_annotations.coco.json')
    with open(output_annotations_file, 'w') as f:
        json.dump(new_annotations, f, indent=4)

    print(f"Novo arquivo de anotações COCO salvo em: {output_annotations_file}")

# Parâmetros de entrada
input_folder = '../dataset/processed/test/images'
output_folder = '../dataset/rois/test'
coco_annotation_file = '../dataset/processed/test/annotations_coco.json'
fovea_annotation_file = '../dataset/fovea-center/test-fovea-center.json'

# Executa o processo de recorte e ajuste das anotações
process_images_with_annotations(input_folder, output_folder, coco_annotation_file, fovea_annotation_file)