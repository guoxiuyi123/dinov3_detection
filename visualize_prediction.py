import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision.transforms import v2
from dinov3.eval.detection.models.detr import build_model, PostProcess
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.hub.backbones import dinov3_vitb16
from dinov3.eval.detection.util.misc import nested_tensor_from_tensor_list

# ================= 配置 =================
# 替换为您刚才用的那张图片路径
IMAGE_PATH = "/home/pc/gxy/dataset/coco2017/coco/images/val2017/000000000139.jpg" 
CHECKPOINT_PATH = "/home/pc/gxy/DINO/dinov3_detection/output_checkpoints/checkpoint_epoch_0.pth"
# =======================================

def main():
    device = torch.device("cuda")
    
    # 1. 加载模型
    print("Building model...")
    config = DetectionHeadConfig(
        num_classes=91, hidden_dim=768, backbone_use_layernorm=False,
        layers_to_use=[2, 5, 8, 11], n_windows_sqrt=2, proposal_feature_levels=4,
        proposal_in_stride=16, proposal_tgt_strides=[8, 16, 32, 64],
        with_box_refine=True, two_stage=True
    )
    backbone = dinov3_vitb16(pretrained=False)
    model = build_model(backbone, config)
    model.to(device)
    
    # 2. 加载权重
    print(f"Loading weights from {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ 错误：找不到权重文件！")
        return
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. 预处理图片
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误：找不到图片 {IMAGE_PATH}")
        return

    # 使用 OpenCV 读取原图用于画图 (BGR)
    img_cv = cv2.imread(IMAGE_PATH)
    orig_h, orig_w = img_cv.shape[:2]
    
    # 使用 PIL 读取用于推理 (RGB)
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(800, max_size=1333, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img_pil).to(device)
    nested = nested_tensor_from_tensor_list([img_tensor]).to(device)

    # 4. 推理
    print("Inference...")
    with torch.no_grad():
        outputs = model(nested)

    # 5. 后处理
    postprocessor = PostProcess()
    # 注意：这里传入的是原图尺寸 [h, w]
    target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
    results = postprocessor(outputs, target_sizes)[0]

    # 6. 画图与调试
    scores = results['scores'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()

    count = 0
    # 阈值设为 0.03
    CONF_THRESHOLD = 0.3
    
    print(f"\n🔍 --- DEBUG INFO ---")
    print(f"Original Image Size: {orig_w} x {orig_h}")
    
    for i, (score, box) in enumerate(zip(scores, boxes)):
        if score > CONF_THRESHOLD:
            count += 1
            
            # 获取坐标
            x1, y1, x2, y2 = box
            
            # --- 自动修复逻辑 ---
            # 如果坐标小于 1.0 (说明是归一化坐标)，手动乘回去
            if x2 <= 1.5 and y2 <= 1.5: 
                print(f"⚠️ 检测到归一化坐标 (0-1)，正在自动修正...")
                x1 *= orig_w
                x2 *= orig_w
                y1 *= orig_h
                y2 *= orig_h
            # -------------------
            
            # 转为整数
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            
            # 打印前几个框的坐标，看看是否正常
            if count <= 5:
                print(f"Object {count}: Score={score:.4f}, Box=[{ix1}, {iy1}, {ix2}, {iy2}]")

            # 画框 (红色，线条加粗到 3)
            cv2.rectangle(img_cv, (ix1, iy1), (ix2, iy2), (0, 0, 255), 3)
            cv2.putText(img_cv, f"{score:.2f}", (ix1, iy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    save_path = "debug_result.jpg"
    cv2.imwrite(save_path, img_cv)
    print(f"---------------------")
    print(f"✅ 处理完成！检测到 {count} 个目标。")
    print(f"📸 结果已保存为: {save_path} (请打开查看)")

if __name__ == "__main__":
    main()