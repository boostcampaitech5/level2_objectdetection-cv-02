## Yolo V6 3.0 모델을 학습하기 위해 필요한 코드
- - -
- [Yolo v6](https://github.com/meituan/YOLOv6)
- custom_dataset.py
    - Yolo v6에서 요구하는 형식대로 디렉토리를 만들고, 데이터를 옮겨줄 때 사용하는 코드입니다.
- recycle.yaml
    - Yolo v6를 학습할 때 필요한 yaml 파일로, 데이터셋의 경로 및 class에 대한 정보가 담겨있습니다.
- submission.py
    - 학습한 모델을 바탕으로, LB에 제출할 csv 파일을 생성해주는 함수입니다.