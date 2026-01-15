# 数据加载
import torch
import torchvision.transforms.functional as F
from torchvision.datasets import CocoDetection
from dinov3.eval.detection.util.misc import NestedTensor

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class CocoDetectionWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def prepare(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        if isinstance(image_id, int): image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno] 
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 0] += boxes[:, 2] / 2
            boxes[:, 1] += boxes[:, 3] / 2
            image_size = torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = boxes / image_size
            boxes.clamp_(min=0.0, max=1.0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = [max(s) for s in zip(*[img.shape for img in tensor_list])]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def make_transforms(image_set):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return normalize