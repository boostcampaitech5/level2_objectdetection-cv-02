import torch

from tqdm import tqdm

from utils import calculate_mAP

# TODO: test_loader가 batch 단위로 들어왔을 때 에러 해결
def evaluate(test_loader, model, cfgs, device):
    """모델 성능을 평가할 때 사용하는 함수.

    Args:
        test_loader (_type_): 평가 데이터셋의 dataloader. e.g., validation, test
        model (_type_): 성능을 평가할 object detection 모델
        cfgs (dict): 사전에 정의한 yaml 파일을 불러온 dict 형식의 설정 값들.
        device (_type_): 학습에 사용할 장비 e.g., cuda, cpu

    Returns:
        _type_: mAP(클래스 별 AP를 평균낸 결과), APs(클래스 별 AP)
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = []
        for images, targets, image_ids in tqdm(test_loader, desc='Evaluating'):
            images = list(image.float().to(device) for image in images)
           
            output = model(images)

            for image_id, out in zip(image_ids, output):
                outputs.append({'image_id': image_id.tolist(),
                                'boxes': out['boxes'].tolist(),
                                'scores': out['scores'].tolist(),
                                'labels': out['labels'].tolist()})
            

        # Calculate mAP
        mean_ap, ap = calculate_mAP(outputs, cfgs['path']['valid_annotation'], cfgs['valid']['score_threshold'])
    
    return mean_ap, ap


# Test Code
if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from utils import get_device
    from my_dataset import CustomDataset
    from transform import get_valid_transform

    device = get_device()
    dataset = CustomDataset("/opt/ml/dataset/train.json", "/opt/ml/dataset", get_valid_transform())
    dataloader = DataLoader(dataset)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    evaluate(dataloader, model, device)