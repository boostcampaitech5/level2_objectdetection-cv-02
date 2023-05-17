# TODO: inference한 결과 txt 파일을 불러와서 원래 좌표계로 만들어주고
# (center_x, center_y, width, height) -> (xmin, ymin, xmax, ymax)
# 저장
from pycocotools.coco import COCO
import pandas as pd

import os
import argparse
import datetime


def get_save_folder_name():
    """파일을 날짜별로 폴더에 저장하고 싶을 때, 폴더명을 만들어주는 함수

    Returns:
       str: "2023-05-12"와 같이 폴더명으로 사용할 문자열 반환
    """
    today = datetime.datetime.now()
    save_folder_name = f"{today.year}-{today.month}-{today.day}"

    return save_folder_name


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Inference based on YOLO v6')

    # parser
    parser.add_argument('--test_json_path', type=str, default='../dataset/test.json', help='select test json path (default : ../dataset/test.json)')
    parser.add_argument('--label_path', type=str, default='./runs/inference/yolov6l6/labels', help='select config path (default : ./runs/inference/yolov6l6/labels)')

    args = parser.parse_args()

    return args


def make_submission(args):
    prediction_strings = []
    file_names = []
    coco = COCO(args.test_json_path)
    
    label_file_names = os.listdir(args.label_path)
    label_file_names.sort(key=lambda x: x.split('.')[0][4:].zfill(4))

    for img_id in coco.getImgIds():
        prediction_string = ''

        label_path = os.path.join(args.label_path, 'test' + str(img_id) + '.txt')
        img_info = coco.loadImgs(img_id)[0]

        # 모델이 예측한 결과가 없는 경우
        if not os.path.exists(label_path):
            prediction_strings.append(prediction_string)
            file_names.append(img_info['file_name'])
            continue

        # 예측한 결과가 있다면 pascal voc 형태로 변환 및 submission.csv 생성
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            category_id, bbox, score = line[0], [float(x) for x in line[1:-1]], float(line[-1])

            center_x, center_y, width, height = bbox
            center_x, center_y = center_x * 1024, center_y * 1024
            width, height = width * 1024, height * 1024

            x_min = center_x - (width / 2)
            y_min = center_y - (height / 2)
            x_max = center_x + (width / 2)
            y_max = center_y + (height / 2)

            prediction_string += str(category_id) + ' ' + str(score) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' '
        
        prediction_strings.append(prediction_string)
        file_names.append(img_info['file_name'])

    # submission 파일 생성
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    save_file_name = args.label_path.split('/')[3]
    save_folder_name = get_save_folder_name()

    print("Save a new submission...", end=' ')
    save_path = f'./submission/{save_folder_name}/{save_file_name}_submission.csv'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    submission.to_csv(save_path, index=None)
    print(submission.head())
    print("Done!")


if __name__ == "__main__":
    # argument 정의
    args = parse_args()

    # label 경로를 가지고 submission 파일 생성
    make_submission(args)
