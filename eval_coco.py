import os
import sys
import torch
import json
import tqdm
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= 配置 =================
COCO_ROOT = "/home/pc/gxy/dataset/coco2017/coco" 
IMG_PATH = os.path.join(COCO_ROOT, "images/val2017")
ANN_PATH = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")
# 记得训练完改这里：checkpoint_epoch_0.pth
HEAD_PATH = "/home/pc/gxy/DINO/dinov3/dinov3-main/output_checkpoints/checkpoint_epoch_0.pth"
RESULT_JSON = "coco_val_results_b16.json"
REPO_DIR = "/home/pc/gxy/DINO/dinov3/dinov3-main"
if REPO_DIR not in sys.path: sys.path.append(REPO_DIR)
# =======================================

from dinov3.eval.detection.models.detr import build_model, PostProcess
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.hub.backbones import dinov3_vitb16
from dinov3.eval.detection.util.misc import nested_tensor_from_tensor_list

def get_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(800, max_size=1333, antialias=True), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def build_my_detector():
    config = DetectionHeadConfig(
        num_classes=91, 
        hidden_dim=768,
        backbone_use_layernorm=False,
        layers_to_use=[2, 5, 8, 11], 
        n_windows_sqrt=2,
        proposal_feature_levels=4,
        proposal_in_stride=16,
        proposal_tgt_strides=[8, 16, 32, 64],
        with_box_refine=True, 
        two_stage=True, 
    )
    backbone = dinov3_vitb16(pretrained=False) 
    model = build_model(backbone, config)
    return model

def main():
    device = torch.device("cuda")
    print(f"Loading COCO val: {ANN_PATH}")
    dataset = CocoDetection(root=IMG_PATH, annFile=ANN_PATH)
    transform = get_transform()

    model = build_my_detector()
    model.to(device)
    
    print(f"Loading weights: {HEAD_PATH}")
    if not os.path.exists(HEAD_PATH):
        print("Weights not found!")
        return
        
    checkpoint = torch.load(HEAD_PATH, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')} or state_dict
    
    # 手动处理一下可能没有 module 前缀的情况
    if not any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = state_dict
    else:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    results = []
    print("Starting inference...")
    postprocessor = PostProcess()
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            img_pil, _ = dataset[i]
            image_id = dataset.ids[i]
            orig_w, orig_h = img_pil.size
            
            img_tensor = transform(img_pil).to(device)
            nested_inputs = nested_tensor_from_tensor_list([img_tensor]).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(nested_inputs)
            
            target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
            results_dict = postprocessor(outputs, target_sizes)[0]

            boxes = results_dict['boxes'].cpu().numpy()
            scores = results_dict['scores'].cpu().numpy()
            labels = results_dict['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05: continue 
                x1, y1, x2, y2 = box
                res = {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(score)
                }
                results.append(res)

    print(f"Inference done. Valid objects: {len(results)}")
    if len(results) > 0:
        with open(RESULT_JSON, "w") as f: json.dump(results, f)
        cocoGt = COCO(ANN_PATH)
        cocoDt = cocoGt.loadRes(RESULT_JSON)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

if __name__ == "__main__":
    main()