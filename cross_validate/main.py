import os
import random
from sklearn.model_selection import KFold
import shutil

# Função para preparar os folds
def prepare_folds(image_folder, label_folder, output_folder, k_folds=5):
    # Listar arquivos de imagem no diretório especificado
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    # Criar lista de anotações correspondentes
    labels = [os.path.splitext(f)[0] + '.txt' for f in images]

    # Filtrar imagens que possuem arquivos de anotação correspondentes
    images = [img for img in images if os.path.exists(os.path.join(label_folder, os.path.splitext(img)[0] + '.txt'))]

    # Dividir dados em K folds usando validação cruzada
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = list(kf.split(images))

    # Preparar os folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Definir pastas para treino e validação
        fold_train_folder = os.path.join(output_folder, f'fold_{fold_idx}', 'train')
        fold_val_folder = os.path.join(output_folder, f'fold_{fold_idx}', 'val')

        # Criar subpastas para imagens e rótulos
        train_images_folder = os.path.join(fold_train_folder, 'images')
        train_labels_folder = os.path.join(fold_train_folder, 'labels')

        val_images_folder = os.path.join(fold_val_folder, 'images')
        val_labels_folder = os.path.join(fold_val_folder, 'labels')

        # Criar diretórios para armazenar dados dos folds
        os.makedirs(train_images_folder, exist_ok=True)
        os.makedirs(train_labels_folder, exist_ok=True)
        os.makedirs(val_images_folder, exist_ok=True)
        os.makedirs(val_labels_folder, exist_ok=True)

        # Copiar arquivos de treino para as pastas correspondentes
        for idx in train_idx:
            image_name = images[idx]
            label_name = os.path.splitext(image_name)[0] + '.txt'

            shutil.copy(os.path.join(image_folder, image_name), train_images_folder)
            shutil.copy(os.path.join(label_folder, label_name), train_labels_folder)

        # Copiar arquivos de validação para as pastas correspondentes
        for idx in val_idx:
            image_name = images[idx]
            label_name = os.path.splitext(image_name)[0] + '.txt'

            shutil.copy(os.path.join(image_folder, image_name), val_images_folder)
            shutil.copy(os.path.join(label_folder, label_name), val_labels_folder)

        # Exibe informações sobre o fold atual
        print(f"Fold {fold_idx}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")

# Função principal para inicializar o processo
def main():
    # Definir caminhos das pastas de entrada e saída
    image_folder = '../dataset/processed/train/images'
    label_folder = '../dataset/processed/train/labels'
    output_folder = '../dataset/k_folds'

    # Definir o número de folds
    k_folds = 5

    # Criar o diretório de saída, se não existir
    if not os.path.exists(output_folder):
      os.mkdir(output_folder)

    # Criar os folds com base nos dados fornecidos
    prepare_folds(image_folder, label_folder, output_folder, k_folds)

# Executar o script se for chamado diretamente
if __name__ == "__main__":
    main()
