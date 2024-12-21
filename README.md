## Trabalho para cadeira de Aprendizado de máquina do PPGC
Implementação de um pipeline para extração de regiões de interesse em imagens de fundo de olho, utilizando a mácula como região alvo.

## Instalação
```
conda create -n yolov10 python=3.9
conda activate yolov11
pip install -r requirements.txt
pip install -e .
```

## Pré-processamento
Remoção parcial do fundo preto
```
cd pre_processing/cropping
python main.py
cd ..
```
Aplicação dos filtros de imagem
```
cd filters
python main.py
cd ../..
```
## Conversão das anotações para o formato YOLO
```
cd pre_processing/JSON2YOLO
python general_json2yolo.py
cd ../..
```
## Geração dos K-folds
```
python ./cross_validate/main.py
cd ..
```
## YOLOV11
Acessar a pasta do modelo
```
cd yolov11
```
### Treinamento
Ajustar o dataset dentro do arquivo coco.yaml
```
python train.py
```
### Gerar as coordenadas das classes
```
python predict.py
```

## Autores
* **Marcelo Dias** - Programa de pós-graduação em ciêncida da computação, Universidade Federal de Pelotas, Brasil.
* **Leandro Tavares** - Programa de pós-graduação em ciêncida da computação, Universidade Federal de Pelotas, Brasil.
