import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataloader.dataset import LungNoduleDataset
from model.diffusion import Diffusion


cfg = DictConfig({
    "timesteps": 1000,
    "sampling_timesteps": 1000,
    "beta_schedule": "linear",
    "schedule_fn_kwargs": {},
    "objective": "pred_noise",
    "use_fused_snr": False,
    "snr_clip": 10,
    "cum_snr_decay": 0.9,
    "ddim_sampling_eta": 0.0,
    "clip_noise": 1.0,
    "architecture": {
        "network_size": 128,
        "dim_mults": [1, 2, 4],
        "attn_resolutions": [16],     # 注意力作用的分辨率
        "attn_dim_head": 64,
        "attn_heads": 4,
        "use_linear_attn": False,
        "use_init_temporal_attn": True,
        "time_emb_type": "sinusoidal",
        "resolution": 64             
    },
    "stabilization_level": 0
})


if __name__ == '__main__':
    # 数据加载部分
    csv_path = '/home/zk/MICCAI/newmainroi.csv'  
    data_dir = '/home/zk/MICCAI/roiresize'

    # 加载 CSV 数据，划分训练集和验证集
    csv_data = pd.read_csv(csv_path)
    subject_ids = csv_data['Subject ID'].unique()
    train_ids, val_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)
    train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
    val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]
        
    train_dataset = LungNoduleDataset(train_data, data_dir, normalize=True)
    val_dataset = LungNoduleDataset(val_data, data_dir, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Diffusion(x_shape=(1, 16, 64, 64), external_cond_dim=0, is_causal=True, cfg=cfg).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            # 获取 T2 视频数据，形状 [B, C, F, H, W]
            t2, label = batch[2].to(device), batch[3].to(device)  

            # 随机生成时间步 t
            noise_levels = torch.randint(
                low=0, high=model.timesteps, size=(t2.size(0),), device=device
            ).unsqueeze(-1)  # 确保形状为 [batch_size, 1]

            # 给 T2 添加噪声
            noisy_t2 = model.q_sample(x_start=t2, t=noise_levels)  # 添加噪声，形状 [B, C, F, H, W]

            # 前向传播：通过模型预测去噪
            pred_t2, loss = model(
                x=noisy_t2,
                external_cond=None,  # 无外部条件
                noise_levels=noise_levels
            )

            # 计算损失（模型输出与真实 T2 的均方误差）
            loss = criterion(pred_t2, t2)

            # 优化步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

            # 打印训练状态
            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 打印每轮的平均损失
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss/len(train_loader):.4f}")

    # 验证模型
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for step, batch in enumerate(val_loader):
            t2, label = batch[2].to(device), batch[3].to(device)

            # 随机生成时间步 t
            noise_levels = torch.randint(
                low=0, high=model.timesteps, size=(t2.size(0),), device=device
            ).unsqueeze(-1)  # 确保形状为 [batch_size, 1]

            # 给 T2 添加噪声
            noisy_t2 = model.q_sample(x_start=t2, t=noise_levels)  # 添加噪声，形状 [B, C, F, H, W]

            # 前向传播：通过模型预测去噪
            pred_t2, loss = model(
                x=noisy_t2,
                external_cond=None,
                noise_levels=noise_levels
            )

            # 计算验证损失
            loss = criterion(pred_t2, t2)
            val_loss += loss.item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")





