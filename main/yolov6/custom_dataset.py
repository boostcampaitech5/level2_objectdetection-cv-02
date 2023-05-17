from pycocotools.coco import COCO

import argparse
import os
from tqdm import tqdm
from shutil import copyfile


def parse_args():
    
    parser = argparse.ArgumentParser(description='formatting custom dataset to run YOLO V6')

    # parser
    parser.add_argument('--train_json_path', type=str, default='/opt/ml/json/detectron2_train.json', help='select train json path (default : /opt/ml/json/detectron2_train.json)')
    parser.add_argument('--val_json_path', type=str, default='/opt/ml/json/detectron2_val.json', help='select validation json path (default : /opt/ml/json/detectron2_val.json)')
    parser.add_argument('--test_json_path', type=str, default='/opt/ml/dataset/test.json', help='select test json path (default : /opt/ml/dataset/test.json)')

    args = parser.parse_args()

    return args


def make_yolov6_dataset_dir(root='/opt/ml'):
    """root 폴더를 기준으로, yolo v6에서 요구하는 형식에 맞춰 디렉토리를 생성해주는 함수입니다.

    Args:
        root (str, optional): root 경로입니다. Defaults to '/opt/ml'.
    """
    base_path = os.path.join(root, 'custom_dataset')

    dir_path = []
    dir_path.append(os.path.join(base_path, 'images/train'))
    dir_path.append(os.path.join(base_path, 'images/val'))
    dir_path.append(os.path.join(base_path, 'images/test'))

    dir_path.append(os.path.join(base_path, 'labels/train'))
    dir_path.append(os.path.join(base_path, 'labels/val'))
    dir_path.append(os.path.join(base_path, 'labels/test'))

    for path in dir_path:
        if not os.path.exists(path):
            os.makedirs(path)

    return dir_path


def yolo_v6_formatting(coco_instance, dir_path, mode='train'):
    """pycocotools.coco.COCO 객체를 불러와, YOLO v6 모델을 학습시키기 위한 형식으로 변경합니다.

    Args:
        coco_instance (_type_): pycocotools.coco.COCO 객체입니다.
    """
    # json에서 가져온 정보를 저장하기 위해, 경로들을 설정합니다.
    train_dataset_path = '/opt/ml/dataset/'
    test_dataset_path = '/opt/ml/dataset/'

    train_imgs_path = dir_path[0]
    val_imgs_path = dir_path[1]
    test_imgs_path = dir_path[2]

    train_labels_path = dir_path[3]
    val_labels_path = dir_path[4]

    # 전체 img id를 가져옵니다.
    img_ids = coco_instance.getImgIds()

    # 가져온 img id를 바탕으로, 포맷팅을 수행합니다.
    for img_id in tqdm(img_ids):
        img_info = coco_instance.loadImgs(img_id)[0] # img 정보를 가져옵니다.
        
        # 기존 이미지가 저장되어있는 경로로부터, 새로운 경로에 이미지를 복사해서 저장합니다.
        if mode == 'train':
            orig_img_path = os.path.join(train_dataset_path, img_info['file_name'])
            new_img_path = os.path.join(train_imgs_path, 'train' + str(img_id) + '.jpg')

            copyfile(orig_img_path, new_img_path)

            # 이미지는 복사해서 저장했으니, label.txt를 만들어야 합니다.
            # 우선 img_info에 저장되어있는 annotation ids를 가져옵니다.
            ann_ids = coco_instance.getAnnIds(imgIds=img_info['id'])

            # ann_ids로부터 annotations을 가져옵니다.
            anns = coco_instance.loadAnns(ann_ids)

            # category id와, bbox를 normalize해서 가져옵니다.
            yolo_v6_anns = []
            for ann in anns:
                category_id = ann['category_id']
                x_min, y_min, width, height = ann['bbox'] # coco 형식의 bbox 정보입니다.

                center_x = (x_min + (width / 2)) / 1024 # 중심점을 구한 뒤, 0 ~ 1 사이의 값을 갖도록 normalize 합니다.
                center_y = (y_min + (height / 2)) / 1024 # y도 마찬가지입니다.

                width, height = width / 1024, height / 1024 # width, height도 normalize 해줍니다.

                # 띄어쓰기를 기준으로 결합한 뒤 추가해줍니다.
                yolo_v6_label = ' '.join([str(category_id), str(center_x), str(center_y),
                                          str(width), str(height)])
                yolo_v6_anns.append(yolo_v6_label)


            # label 경로에 새로운 파일을 만듭니다.
            label_path = os.path.join(train_labels_path, 'train' + str(img_id) + '.txt')
            with open(label_path, 'w') as f:
                for yolo_v6_ann in yolo_v6_anns:
                    f.write(yolo_v6_ann + '\n')


        elif mode == 'val':
            orig_img_path = os.path.join(train_dataset_path, img_info['file_name'])
            new_img_path = os.path.join(val_imgs_path, 'val' + str(img_id) + '.jpg')

            copyfile(orig_img_path, new_img_path)

            # 이미지는 복사해서 저장했으니, label.txt를 만들어야 합니다.
            # 우선 img_info에 저장되어있는 annotation ids를 가져옵니다.
            ann_ids = coco_instance.getAnnIds(imgIds=img_info['id'])

            # ann_ids로부터 annotations을 가져옵니다.
            anns = coco_instance.loadAnns(ann_ids)

            # category id와, bbox를 normalize해서 가져옵니다.
            yolo_v6_anns = []
            for ann in anns:
                category_id = ann['category_id']
                x_min, y_min, width, height = ann['bbox'] # coco 형식의 bbox 정보입니다.

                center_x = (x_min + (width / 2)) / 1024 # 중심점을 구한 뒤, 0 ~ 1 사이의 값을 갖도록 normalize 합니다.
                center_y = (y_min + (height / 2)) / 1024 # y도 마찬가지입니다.

                width, height = width / 1024, height / 1024 # width, height도 normalize 해줍니다.

                # 띄어쓰기를 기준으로 결합한 뒤 추가해줍니다.
                yolo_v6_label = ' '.join([str(category_id), str(center_x), str(center_y),
                                          str(width), str(height)])
                yolo_v6_anns.append(yolo_v6_label)


            # label 경로에 새로운 파일을 만듭니다.
            label_path = os.path.join(val_labels_path, 'val' + str(img_id) + '.txt')
            with open(label_path, 'w') as f:
                for yolo_v6_ann in yolo_v6_anns:
                    f.write(yolo_v6_ann + '\n')

        elif mode == 'test':
            orig_img_path = os.path.join(test_dataset_path, img_info['file_name'])
            new_img_path = os.path.join(test_imgs_path, 'test' + str(img_id) + '.jpg')

            copyfile(orig_img_path, new_img_path)
        
        else:
            print('Train 혹은 Validation만 지원합니다.')
            raise ValueError
    

if __name__ == "__main__":
    # 1. train, validation data의 json 경로를 입력받습니다.
    args = parse_args()

    # 2. train, val json을 불러옵니다.
    train_coco = COCO(args.train_json_path)
    val_coco = COCO(args.val_json_path)
    test_coco = COCO(args.test_json_path)

    # 3. YOLO v6에서 요구하는 방식대로 디렉토리를 생성합니다.
    dir_path = make_yolov6_dataset_dir()

    # 4. 생성한 디렉토리에 YOLO v6에서 요구하는 형식대로 데이터를 저장합니다.
    yolo_v6_formatting(train_coco, dir_path, mode='train')
    yolo_v6_formatting(val_coco, dir_path, mode='val')
    yolo_v6_formatting(test_coco, dir_path, mode='test')

