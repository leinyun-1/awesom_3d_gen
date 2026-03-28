# Direct3D 论文解读：首个可扩展的原生3D生成

> 来源：知乎
> 原文链接：https://zhuanlan.zhihu.com/p/705647869

## 论文定位

Direct3D 由南京大学/DreamTech AI 提出，是**首个可扩展到wild输入图像的原生3D生成模型**，无需多视图扩散模型或SDS优化。

## 核心创新：D3D-VAE

Direct3D 的关键贡献在于提出了 **D3D-VAE**：

1. **Triplane NeRF 编码器**：将3D shape编码为triplane表示
2. **Token化**：将triplane转化为可处理的token序列
3. **DiT训练**：在token latent上训练DiT，使用image作为condition

### 与3DShape2VecSet的区别

| 维度 | 3DShape2VecSet | Direct3D |
|------|---------------|----------|
| 输入 | 点云/mesh | **Wild images** |
| Latent | 1D tokens (无结构) | Triplane tokens (半结构化) |
| 扩散 | DDPM | EDM |
| 可扩展性 | 有限 | **高** |
| condition | 无/有限 | Image cross-attention |

## Direct3D → Direct3D-S2 的演进

Direct3D-S2 是直接的升级版，核心改进在VAE：

1. **从Triplane → 稀疏SDF体素**：直接处理3D SDF volume
2. **SS-VAE (Sparse SDF VAE)**：稀疏卷积 + Transformer 的对称encoder-decoder
3. **SSA (Spatial Sparse Attention)**：仅在有内容的稀疏位置上计算attention
4. **分辨率突破**：从Direct3D的中等分辨率跃升至**1024³**
5. **效率突破**：仅需8 GPU即可训练

### 技术细节

- Encoder：稀疏3D卷积逐步降采样 → Transformer处理全局关系
- Decoder：Transformer → 稀疏3D反卷积逐步上采样
- 损失函数：SDF重建损失 + KL散度

## 学术地位

Direct3D 系列代表了南京大学在原生3D生成领域的持续深耕：
- Direct3D 定义了"image → 3D latent → DiT → 3D"的简洁管线
- Direct3D-S2 在VAE端实现了质的飞跃（稀疏卷积 + 高模输入输出）
- 其SS-VAE范式可能成为后续3D VAE的标准方案之一
