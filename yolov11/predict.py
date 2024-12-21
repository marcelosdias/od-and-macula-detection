from ultralytics import YOLO

import os

for dataset in ['idrid']:
    model = YOLO('./model.pt')
    input_folder = f'../dataset/processed/test/images'

    files = os.listdir(input_folder)

    for file in files:
      result = model(
        source=f'{input_folder}/{file}',
        mode='predict',
        save=True,
        save_txt=True,
        conf=0.7,
      )