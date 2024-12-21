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
python ./pre_processing/cropping/main.py
```
Aplicação dos filtros de imagem
```
python ./pre_processing/filters/main.py
```
Conversão das anotações para o formato YOLO
```
python ./JSON2YOLO/filters/general_json2yolo.py
```
Geração dos K-folds
```
python ./cross_validate/main.py
```
Treinamento do modelo
```
python ./model/train.py
```
Teste do modelo
```
python ./model/test.py
```
Predição com o modelo
```
python ./model/predict.py
```

## Autores
* **Marcelo Dias** - Programa de pós-graduação em ciêncida da computação, Universidade Federal de Pelotas, Brasil.
* **Leandro Tavares** - Programa de pós-graduação em ciêncida da computação, Universidade Federal de Pelotas, Brasil.
