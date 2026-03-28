# TRELLIS 技术解读：结构化3D Latent

> 来源：知乎（多篇文章综合整理）
> 原文链接：
> - https://zhuanlan.zhihu.com/p/30989191234
> - https://zhuanlan.zhihu.com/p/1941395733052920217

## 核心问题

在 TRELLIS 之前，3D 生成面临一个根本性矛盾：

- **3DShape2VecSet** 等方法将 3D shape 编码为一组 **无空间结构的 latent tokens**（1D 序列）
- 这导致 latent 缺乏空间位置信息，不利于空间编辑，且生成质量受限

## TRELLIS 的解决方案：Structured LATent (SLAT)

TRELLIS 提出 **结构化潜空间 (SLAT)**：

1. **特征体素作为 latent**：每个 latent token 对应一个 3D 空间位置上的 feature voxel
2. **位置 + 特征双重编码**：latent 天然具有 3D 空间结构
3. **多种输出格式**：同一个 SLAT 可以 decode 为 mesh、3D Gaussian、radiance field 等
4. **Rectified Flow** 替代 DDPM：训练更稳定，采样更高效

## 与 LDM 的类比

| 2D LDM | TRELLIS |
|--------|---------|
| Image → VAE → 2D latent | Multi-view → Encoder → SLAT (3D structured latent) |
| 2D latent 上训练扩散 | SLAT 上训练 Rectified Flow |
| 2D latent → VAE decoder → Image | SLAT → Decoder → Mesh/3DGS/NeRF |

## 关键技术细节

- **编码器**：从多视图图像编码为 SLAT（利用 3D 空间对应关系）
- **稀疏结构**：只在有内容的 voxel 位置上保存 latent（节省计算）
- **解码灵活性**：使用不同 decoder head 输出不同 3D 表示

## 影响

TRELLIS 的结构化 latent 思想深刻影响了后续工作：
- TRELLIS.2 进一步引入 O-Voxel，实现原生 3D VAE
- Direct3D-S2 虽然使用稀疏卷积方案，但也保留了空间结构
- Hunyuan3D 系列在纹理生成中也借鉴了结构化表示的思想
