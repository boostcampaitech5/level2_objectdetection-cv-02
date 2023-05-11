import copy
import torch

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

def MyMapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.Resize((1024, 1024)),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict


def InferenceMapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict