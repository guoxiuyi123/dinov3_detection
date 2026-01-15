import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import nn
from .utils import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 获取模型输出
        # sigmoid() 可能导致数值不稳定，我们先转换，如果有 NaN 则替换
        out_prob = outputs["pred_logits"].flatten(0, 1).float().sigmoid()  
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float()  

        # --- [安全检查 1] 检查模型输出是否已经坏了 ---
        if torch.isnan(out_prob).any() or torch.isinf(out_prob).any():
            print("⚠️ 警告: 模型预测的 Class Logits 包含 NaN/Inf！这通常意味着发生了梯度爆炸。")
            # 临时修复：将其置为 0，防止崩溃，但需要检查 LR
            out_prob = torch.nan_to_num(out_prob, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(out_bbox).any() or torch.isinf(out_bbox).any():
            print("⚠️ 警告: 模型预测的 Boxes 包含 NaN/Inf！")
            out_bbox = torch.nan_to_num(out_bbox, nan=0.5, posinf=1.0, neginf=0.0)

        # 2. 准备 Targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 3. 计算 Cost Matrix
        # Classification Cost
        alpha = 0.25
        gamma = 2.0
        # 添加 1e-8 防止 log(0)
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Box Cost (L1)
        # cdist 对 NaN 非常敏感，确保输入干净
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU Cost
        # 确保 bbox 格式正确
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final Cost
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # --- [安全检查 2] 检查 Cost Matrix 是否有非法值 ---
        # 如果 C 中有 NaN，linear_sum_assignment 会报错
        if torch.isnan(C).any() or torch.isinf(C).any():
            print("⚠️ 警告: Cost Matrix 包含 NaN/Inf！正在尝试自动修复以避免崩溃...")
            # 将 NaN 替换为一个非常大的数，意味着“不想匹配这个”
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

        # 4. 执行匈牙利匹配
        sizes = [len(v["boxes"]) for v in targets]
        
        # 即使修复了，linear_sum_assignment 仍可能因为极端数值报错，加一层 try-catch
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            try:
                # 转换为 numpy 确保兼容性
                c_np = c[i].numpy()
                # 二次清洗：确保 numpy 数组也没有 NaN
                if np.isnan(c_np).any() or np.isinf(c_np).any():
                    c_np = np.nan_to_num(c_np, nan=1e6, posinf=1e6, neginf=-1e6)
                    
                ind = linear_sum_assignment(c_np)
                indices.append((torch.as_tensor(ind[0], dtype=torch.int64), torch.as_tensor(ind[1], dtype=torch.int64)))
            except ValueError as e:
                print(f"❌ 错误: Hungarian Matcher 在第 {i} 个样本处失败: {e}")
                print(f"  Cost Matrix Stats: Min={c_np.min()}, Max={c_np.max()}, HasNaN={np.isnan(c_np).any()}")
                # 遇到无法解决的错误，为了不中断训练，返回空匹配（但这会导致 Loss 很大）
                # 或者直接让它报错停止，便于排查
                raise e 

        return indices