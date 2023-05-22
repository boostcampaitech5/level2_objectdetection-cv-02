## **재활용 품목 분류를 위한 Object Detection (Naver Boostcamp AI Tech CV-02조 팀 멋쟁이)**

### 📌 **대회 정보**
- - -
- **대회 주제** : 주어진 사진에서 쓰레기를 Detection하는 모델 구현
- **대회 목표**
    - 체계적인 실험 관리 (e.g., 비교분석을 위한 table 작성)
    - robust한 모델 설계 (e.g., train/test data에 대한 성능차이가 작은 모델 설계)
    - 적극적인 GitHub 활용을 통한 협업 진행 (e.g., GitHub flow 활용)
- **대회 일정** : 23.05.03 ~ 23.05.18 19:00 (2주)

### 🐦 **Members**
- - -
|**이름**|**역할**|**github**|
|:-:|:-:|:-:|
|김성한|Detectron2 (cascade, tridentnet, faster rcnn, retinanet) 실험, Ensemble|[Happy-ryan](https://github.com/Happy-ryan)|
|박수영|Detectron2, Torchvision Faster R-CNN 실험, Yolo v6 실험, mAP metric 분석|[nstalways](https://github.com/nstalways)|
|이다현|Mmdetection baseline 구성 및 실험, Pseudo labeling/Ensemble 실험|[Eumgil98](https://github.com/Eumgill98)|
|이채원|Mmdetection training baseline 구성 및 실험, 모델 Backbone 및 TTA 실험|[Chaewon829](https://github.com/Chaewon829)|
|정호찬|Detectron2 실험, MMdetection-Cascade Swin L RCNN 실험, Augmentation 실험|[DaHyeonnn](https://github.com/DaHyeonnn)|

### 📝 **Dataset 개요**
- - -
- **전체 데이터셋 통계**
    - 전체 이미지 개수 : 9754 장 **(train 4883, validation 4871)**
    - 클래스 종류 : 10 개 (General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing)
    - 이미지 크기 : (1024, 1024)

- **이미지 예시**<br>
<img src="./data_example.png" width="50%" height="50%"/><br>
**위 이미지는 예시일 뿐이며, 실제 데이터와는 관련이 없습니다.**

- **(주의) Submission & Annotation format**
    - Submission format은 PASCAL VOC 형태!
    - Annotation format은 COCO 형태!
    - **format마다 bbox를 정의하는 방식이 다르므로**, metric 계산 시 주의!! [(Ref)](https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5)

### 🐤 **폴더 구조**
- - -
```
main
├── detectron2
│   ├── tridentnet : detectron2에서 제공하지 않은 모델 사용하기 위한 dir
│   ├── inference.py : model inference 및 submission file 생성
│   ├── mapper.py : data augmentation 담당하는 코드
│   ├── trainer.py : data loader 및 evaluator 생성하는 코드
│   ├── utils.py : config 설정 코드
│   └── train.py : 학습 실행하는 Command Line Interface
│
├── mmdetection
│   ├── augmentation
│   │   ├── BaseAugmentation.py : bbox annotation load 및 tensor 변환만 포함한 Base Aug
│   │   └── CustomAugmentation.py : custonAugmentation을 구성하고 pipeline에 import하는 코드
│   ├── pipeline.py : train, val, test의 Transform pipeline 구성
│   ├── inference.py : model inference 및 submission file 생성
│   └── train.py : 학습 실행하는 Command Line Interface
│    
├── torchvision
│   ├── configs : train/evaluation/inference 시 사용하는 yaml 파일들을 모아둔 폴더
│   ├── model : custom model 코드들을 모아둔 폴더
│   ├── trainer : 모델 별로 train 시 사용하는 코드들을 모아둔 폴더
│   ├── evaluation.py
│   ├── inference.py
│   ├── my_dataset.py
│   ├── my_optimizer.py
│   ├── train.py
│   ├── transform.py
│   └── utils.py 
│
└── yolov6
    ├── custom_dataset.py : yolov6에서 요구하는 형식에 맞게 디렉토리를 재구성하는 코드
    ├── recycle.yaml : 데이터셋의 경로 및 class에 대한 정보가 담겨있는 yaml 파일
    └── submission.py
```

### 🐧 **최종 결과**
- - -
```
Model
├── 2 Stage Model
│   ├── Faster RCNN :0.5385
│   ├── Cascade RCNN :0.5747
│   └── DETR : 0.3987
└── 1 Stage Model
    ├── PAA : 0.5787
    ├── UniverseNet :0.6383
    ├── RetinaNet : 0.3406
    ├── TOOD : 0.4482
    └── YOLOv6 :0.5424
```