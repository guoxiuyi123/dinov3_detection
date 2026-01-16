import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

from dinov3.eval.detection.models.detr import build_model
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.hub.backbones import dinov3_vitb16

# 导入自定义训练引擎
from train_engine.matcher import HungarianMatcher
from train_engine.criterion import SetCriterion
from train_engine.dataset import CocoDetectionWrapper, collate_fn, make_transforms

# ================= 配置区域 =================
IMG_DIR = "/home/pc/gxy/dataset/coco2017/coco/images/train2017"
ANN_FILE = "/home/pc/gxy/dataset/coco2017/coco/annotations/instances_train2017.json"
PRETRAIN_WEIGHTS = "/home/pc/gxy/DINO/dinov3_detection/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# ===========================================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        print(f"Rank {rank} initialized on GPU {gpu}")
        return gpu, rank
    else:
        print("Not using distributed mode")
        return 0, 0

def cleanup():
    dist.destroy_process_group()

def main():
    gpu_id, rank = setup_distributed()
    device = torch.device(f"cuda:{gpu_id}")
    
    # 1. Config
    config = DetectionHeadConfig(
        num_classes=91, 
        hidden_dim=768,
        backbone_use_layernorm=False,
        layers_to_use=[2, 5, 8, 11], 
        n_windows_sqrt=2,
        proposal_feature_levels=4,
        with_box_refine=True,
        two_stage=True,
    )
    
    # 2. Backbone (手动加载 + 解冻)
    print("🏗️ Building Backbone...")
    backbone = dinov3_vitb16(pretrained=False)
    
    if rank == 0:
        print(f"📥 Loading backbone weights from: {PRETRAIN_WEIGHTS}")
    
    if os.path.exists(PRETRAIN_WEIGHTS):
        checkpoint = torch.load(PRETRAIN_WEIGHTS, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("backbone.", "")
            new_state_dict[k] = v
        msg = backbone.load_state_dict(new_state_dict, strict=False)
        if rank == 0:
            print(f"✅ Weights Loaded. Missing keys: {len(msg.missing_keys)}")
    
    # === 🔓 [核心修改] 解冻 Backbone！ ===
    # 官方策略冻结是为了在大规模集群上微调，在单机上我们需要由于数据量和Batch较小，解冻能加速收敛
    for p in backbone.parameters():
        p.requires_grad = True 
        
    config.proposal_in_stride = 16
    config.proposal_tgt_strides = [8, 16, 32, 64]
    
    # 3. Model
    model = build_model(backbone, config)
    model.to(device)
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    # 4. Loss
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    aux_weight_dict = {}
    for i in range(config.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=91,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=['labels', 'boxes']
    )
    criterion.to(device)

    # 5. Data
    dataset_train = CocoDetectionWrapper(
        img_folder=IMG_DIR,
        ann_file=ANN_FILE,
        transforms=make_transforms("train")
    )
    
    sampler_train = DistributedSampler(dataset_train, shuffle=True)
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=4, 
        sampler=sampler_train,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=4,
        pin_memory=True
    )

    # 6. Optimizer [核心修改：差分学习率]
    # Backbone 用小火(1e-5)，Head 用大火(1e-4)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": 1e-4, 
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5, 
        },
    ]

    base_lr = 1e-4 
    optimizer = torch.optim.AdamW(param_dicts, lr=base_lr, weight_decay=1e-4)

    # 7. Scheduler
    warmup_iters = 1000
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_iters)
    iters_per_epoch = len(data_loader_train)
    milestones = [8 * iters_per_epoch, 9 * iters_per_epoch]
    main_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])

    # 8. Loop
    model.train()
    criterion.train()
    
    start_epoch = 0
    max_epochs = 10
    
    if rank == 0:
        print(f"🚀 Start training. Backbone: UNFREEXED (LR 1e-5). Head LR: 1e-4.")

    for epoch in range(start_epoch, max_epochs):
        sampler_train.set_epoch(epoch)
        if rank == 0:
            print(f"------ Epoch {epoch} ------")
            
        for i, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if not torch.isfinite(losses):
                print(f"Loss is {losses}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            optimizer.step()
            lr_scheduler.step()

            if rank == 0 and i % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                l_bbox = loss_dict.get('loss_bbox', torch.tensor(0)).item()
                l_ce = loss_dict.get('loss_ce', torch.tensor(0)).item()
                print(f"Epoch {epoch} | Iter {i}/{iters_per_epoch} | Loss: {losses.item():.2f} (Box:{l_bbox:.2f} Cls:{l_ce:.2f}) | LR: {current_lr:.8f}")

        if rank == 0:
            output_dir = "output_checkpoints"
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    cleanup()

if __name__ == "__main__":
    main()