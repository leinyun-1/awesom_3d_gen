# TRELLIS vs TRELLIS.2 — VAE 与 DiT 设计对比分析

> 以数据流视角（tensor shape变化链）解读两代模型的核心差异。
> 所有维度信息均来自论文原文和附录配置表。

---

## 〇、一句话总结差异

| 维度 | TRELLIS | TRELLIS.2 |
|------|---------|-----------|
| **VAE骨干** | Transformer（3D Swin Attention） | **全稀疏卷积**（类ConvNeXt残差块） |
| **Latent来源** | 从2D多视图图像特征编码而来 | 从**原生3D mesh**直接转换而来 |
| **3D表示** | SLAT（需要SDF场+多视图渲染中间步骤） | **O-Voxel**（mesh↔体素即时双向转换，无需场） |
| **空间压缩率** | ~4×（64³→32³ via sparse conv packing） | **16×**（1024³→64³） |
| **Latent通道数** | 768 | 32 |
| **DiT架构** | 有sparse conv packing + skip connections | **Vanilla DiT**（去掉了packing和skip） |
| **生成阶段** | 两阶段（结构+几何） | **三阶段**（结构+几何+材质PBR） |

---

## 一、VAE 对比

### 1.1 TRELLIS 的 VAE

TRELLIS有**两个独立的VAE**：

#### (A) Sparse Structure VAE — 压缩二值体素网格

```
输入: binary_voxel_grid        [64, 64, 64]         二值体素（0/1）
  │
  ├── 3D Conv (ch=32)          [64, 64, 64, 32]     ← 64³空间，32通道
  ├── Downsample + Conv        [32, 32, 32, 128]    ← 32³空间，128通道
  ├── Downsample + Conv        [16, 16, 16, 512]    ← 16³空间，512通道
  ├── Linear → μ, logσ²       [16, 16, 16, 8]      ← latent: 16³空间，8通道
  │
输出: structure_latent S        [16, 16, 16, 8]      密集3D网格
```

- **编码器 E_S**：3D卷积U-Net，59.3M参数
- **解码器 D_S**：对称U-Net + PixelShuffle上采样，73.7M参数
- **归一化**：LayerNorm（不用GroupNorm）
- **损失**：Dice Loss（因为激活体素很稀疏，正负样本不平衡）

> **注意**：这个VAE处理的是**密集的64³网格**，因为结构信息本身就是dense的占位信息。

#### (B) SLat VAE — 编码结构化特征

```
输入来源: 多视图渲染图像 → DINOv2特征 → 投影到3D体素

输入: voxel_features             [L, C_in]           L个激活体素的聚合特征
      voxel_positions            [L, 3]              每个体素在64³中的坐标
                                                      L ≈ 20000（平均值）

  ├── 加入 sinusoidal 3D位置编码
  ├── 12层 3D Shifted-Window Transformer
  │     每层: 3D-SW-MSA(dim=768, heads=12) → FFN
  │     窗口大小: 8³，层间交替移位
  ├── Linear → μ, logσ²
  │
输出: slat_features z            [L, 768]            结构化latent特征
      slat_positions p           [L, 3]              位置不变，直接保留

组合: SLAT = {(z_i, p_i)}_{i=1}^{L}   每个条目 = 768维特征 + 3D坐标
```

- **编码器 E**：12层 3D Swin Transformer，85.8M参数
- **三个解码器**（共享Transformer骨干，仅输出头不同）：
  - **D_GS** (→3D Gaussians): 每个体素预测K=32个高斯，参数含位置偏移(3) + 颜色(3) + 缩放(3) + 不透明度(1) + 旋转(4) = 14维×32 = 448维/体素
  - **D_RF** (→Radiance Field): 输出CP-decomposition向量，每个体素表示局部8³辐射场
  - **D_M** (→Mesh): Transformer + 2层Sparse Conv Upsampler (64³→256³)，输出FlexiCubes参数(45维) + SDF(8值) + 顶点颜色和法线

> **关键局限**：latent特征来源于**2D图像特征的3D投影聚合**，不是从原生3D数据直接学来的。这意味着：
> 1. 需要先渲染多视图图像
> 2. DINOv2提取2D特征
> 3. 投影到3D体素位置

---

### 1.2 TRELLIS.2 的 VAE

TRELLIS.2 只有**一个统一的SC-VAE**（Sparse Compression VAE），但几何和材质分开编码。

#### 前置步骤：Mesh → O-Voxel（无需神经网络，CPU上几秒完成）

```
输入: triangle_mesh              任意拓扑的三角网格

  ├── 体素化: 找出与mesh相交的体素   → active_voxel_positions  [L, 3]
  ├── 对每个活跃体素计算:
  │     dual_vertex v_i           [3]     ← 通过求解QEF得到的体素内顶点
  │     edge_flags  δ_i           [3]     ← XYZ三条边是否与mesh相交（二值）
  │     split_weight γ_i          [1]     ← 四边形→三角形的剖分权重
  │
输出: o_voxel_shape              [L, 7]   = (v:3 + δ:3 + γ:1)
      positions                  [L, 3]

若含纹理：
  ├── 对每个活跃体素采样表面PBR属性:
  │     base_color c_i            [3]
  │     metallic   m_i            [1]
  │     roughness  r_i            [1]
  │     opacity    α_i            [1]
  │
输出: o_voxel_mat                [L, 6]   = (c:3 + m:1 + r:1 + α:1)
```

> **关键区别**：O-Voxel是从mesh**直接几何计算**得到的，不需要渲染、不需要2D特征提取、不需要SDF场，而且是**可逆的**——从O-Voxel可以直接恢复mesh。

#### SC-VAE Encoder 数据流（以几何编码为例）

```
输入: o_voxel_shape_features     稀疏张量 [L, 6]     L个活跃体素，每个6维
      空间分辨率: 1024³（或512³等，取决于输入）

  ├── Linear(6, 64)              稀疏 [L, 64]        ← 投射到64通道
  │
  ├── ResEnc(64→128) + 2×下采样  稀疏 [L', 128]      ← 空间/2，通道翻倍
  ├── 4× OptResBlock(128)        稀疏 [L', 128]      ← SubMConv3×3 + LayerNorm + MLP(128→512→128) + SiLU
  │
  ├── ResEnc(128→256) + 2×下采样 稀疏 [L'', 256]     ← 空间/4
  ├── 8× OptResBlock(256)        稀疏 [L'', 256]
  │
  ├── ResEnc(256→512) + 2×下采样 稀疏 [L''', 512]    ← 空间/8
  ├── 16× OptResBlock(512)       稀疏 [L''', 512]     ← 这一层最深，16个残差块
  │
  ├── ResEnc(512→1024) + 2×下采样 稀疏 [L'''', 1024] ← 空间/16
  ├── 4× OptResBlock(1024)       稀疏 [L'''', 1024]
  │
  ├── Linear(1024, 64)           稀疏 [L'''', 64]    ← 投射到 32(μ) + 32(logσ²)
  │
输出: shape_latent                稀疏 [L'''', 32]    ← 重参数化采样后
      空间分辨率: 64³（若输入1024³）

具体数值示例（1024³输入）:
  输入层: ~154K 活跃体素 @ 1024³, 6维
  1×下采样后: ~38K @ 512³
  2×下采样后: ~19K @ 256³
  4×下采样后: ~14K @ 128³
  16×下采样后（最终）: ~9.6K @ 64³, 32维
```

- **全部使用稀疏3D卷积**（SubManifold Conv），没有任何Transformer/Self-Attention
- **Optimized Residual Block**：借鉴ConvNeXt思想——单层稀疏卷积 + 宽MLP（通道放大4倍再缩回），比双卷积更强且更高效
- **Sparse Residual Autoencoding Layer (ResEnc)**：下采样时将8个子体素堆叠到通道维度，通过分组平均计算粗残差。作用是缓解高压缩率(16×)下的信息丢失
- **Early-pruning Upsampler**（Decoder中）：上采样前预测二值mask提前剔除无效位置，减少计算

#### SC-VAE Decoder 数据流

```
输入: shape_latent               稀疏 [~9.6K, 32]    @ 64³

  ├── Linear(32, 1024)           稀疏 [~9.6K, 1024]
  ├── 4× OptResBlock(1024)
  │
  ├── Upsample + early-pruning   稀疏 [~14K, 512]    @ 128³
  │     (预测mask剔除无效体素)
  ├── 16× OptResBlock(512)
  │
  ├── Upsample + early-pruning   稀疏 [~19K, 256]    @ 256³
  ├── 8× OptResBlock(256)
  │
  ├── Upsample + early-pruning   稀疏 [~38K, 128]    @ 512³
  ├── 4× OptResBlock(128)
  │
  ├── Upsample + early-pruning   稀疏 [~154K, 64]    @ 1024³
  ├── Linear(64, 7)              稀疏 [~154K, 7]     ← 重建 o_voxel_shape
  │
输出: reconstructed_o_voxel      稀疏 [L_out, 7]     @ 1024³
```

---

### 1.3 VAE 核心差异总结

| 维度 | TRELLIS SLat VAE | TRELLIS.2 SC-VAE |
|------|-----------------|------------------|
| **骨干网络** | 12层 3D Swin Transformer | **全稀疏卷积 U-Net**（32层残差块） |
| **输入来源** | 2D图像 → DINOv2 → 3D投影 | **原生mesh → O-Voxel**（几何计算） |
| **输入维度** | `[L≈20K, C_in]` @ 64³ | `[L≈154K, 6]` @ 1024³ |
| **Latent维度** | `[L≈20K, 768]` @ 64³ | `[L≈9.6K, 32]` @ 64³ |
| **空间压缩** | ~4×（64³→32³打包，DiT中做） | **16×**（1024³→64³，VAE内完成） |
| **通道压缩** | 输入→768（其实是扩展） | 6→32（温和压缩） |
| **Token总数** | ~20K×768 = 1540万参数量级 | **~9.6K×32 = 30万参数量级** |
| **注意力机制** | 3D Shifted Window Self-Attention | **无**（纯卷积） |
| **位置编码** | Sinusoidal 3D | **隐式**（卷积天然保留空间关系） |
| **几何→Mesh** | 需要通过decoder解码SDF等场再提mesh | **O-Voxel直接转mesh**（无需神经网络） |
| **参数量** | E: 85.8M, D: 85-91M | 未明确公布（但卷积网络通常更小） |

> **核心进步**：TRELLIS.2 的 latent 空间比 TRELLIS 紧凑了约 **50倍**（30万 vs 1540万元素），但分辨率从64³提升到了1024³。这全靠两点：(1) 全稀疏卷积的16×空间压缩 (2) 通道数从768降到32。

---

## 二、DiT 对比

### 2.1 TRELLIS 的 DiT

**两阶段生成**：

#### Stage 1: 结构生成器 G_S

```
输入: 
  noise_S                [16, 16, 16, 8]     ← 高斯噪声，与structure_latent同shape
  timestep t             [1]                  ← 标量
  condition              [N_cond, D_cond]     ← CLIP文本特征 或 DINOv2图像特征

处理:
  ├── 展平 16³ → 4096 tokens
  ├── + sinusoidal 3D位置编码          [4096, D_model]
  │
  ├── 重复 N_layers 次:
  │     ├── adaLN(timestep t)          ← 用t调制LayerNorm的scale和shift
  │     ├── MSA (Self-Attention)       ← 全局注意力，QK-Norm(RMSNorm)
  │     ├── MCA (Cross-Attention)      ← condition作为K,V注入
  │     └── FFN
  │
  ├── reshape回3D                      [16, 16, 16, 8]
  │
输出: denoised_S           [16, 16, 16, 8]
  → Decoder D_S → binary_voxel [64, 64, 64]
```

#### Stage 2: SLat特征生成器 G_L

```
输入:
  noise_z                [L, C]              ← L个活跃体素位置上的噪声
  positions p            [L, 3]              ← 来自Stage 1的体素位置
  timestep t             [1]
  condition              [N_cond, D_cond]

处理:
  ├── Sparse Conv Downsampler: 64³ → 32³
  │     L tokens 打包到更少的 L' tokens     ← 缩短序列长度
  │
  ├── + 3D位置编码                          [L', D_model]
  │
  ├── 重复 N_layers/2 次 (Encoder部分):
  │     ├── adaLN(t)
  │     ├── MSA + QK-Norm
  │     ├── MCA (condition注入)
  │     └── FFN
  │     保存 skip connections
  │
  ├── 重复 N_layers/2 次 (Decoder部分):
  │     ├── adaLN(t)
  │     ├── MSA + QK-Norm + skip connection
  │     ├── MCA
  │     └── FFN
  │
  ├── Sparse Conv Upsampler: 32³ → 64³     ← 恢复序列长度
  │
输出: denoised_z           [L, C]            ← C=768(L), 1024(L), 1280(XL)
```

**Large模型配置**：24层, dim=1024, heads=16, G_S=543M参数, G_L=588M参数

### 2.2 TRELLIS.2 的 DiT

**三阶段生成**：

#### 共同架构（三阶段共享同一设计，仅输入输出不同）

```
Vanilla DiT Block（每个Block）:
  ├── AdaLN-single(timestep t)      ← 比adaLN更轻量（参数更少）
  ├── Self-Attention                 ← 全局注意力
  │     + RoPE (Rotary Position Embedding)  ← 比sinusoidal更好的位置泛化
  │     + QK-Norm (RMSNorm)
  ├── Cross-Attention               ← DINOv3-L图像特征作为K,V
  └── FFN (hidden=8192)

配置: 30层, dim=1536, heads=12, MLP宽度=8192
参数量: ~1.3B 每个模型
```

#### Stage 1: 稀疏结构生成

```
输入: noise_structure      密集 [D, D, D, C_s]    ← D=16或32，C_s未明确
      condition            [N_cond, D_cond]       ← DINOv3-L特征

输出: structure_latent     → 解码为活跃体素位置
```

#### Stage 2: 几何Latent生成

```
输入: noise_shape          稀疏 [L≈9.6K, 32]      ← 与SC-VAE latent同shape
      active_positions     [L, 3]                 ← 来自Stage 1
      condition            [N_cond, D_cond]       ← DINOv3-L特征

处理:
  ├── Linear(32, 1536)     [L, 1536]             ← 投射到DiT hidden dim
  ├── + RoPE(positions)
  ├── 30× Vanilla DiT Block
  │     Self-Attn + Cross-Attn + FFN
  │     （无sparse conv packing，无skip connection）
  ├── Linear(1536, 32)     [L, 32]               ← 投射回latent dim

输出: denoised_shape_latent [L≈9.6K, 32]
  → SC-VAE Decoder → O-Voxel shape → Mesh
```

#### Stage 3: 材质Latent生成（TRELLIS.2新增）

```
输入: noise_mat            稀疏 [L, 32]           ← 材质latent噪声
      shape_latent         稀疏 [L, 32]           ← Stage 2的结果
      condition            [N_cond, D_cond]       ← DINOv3-L特征

处理:
  ├── concat(noise_mat, shape_latent) → [L, 64]  ← 通道拼接
  ├── Linear(64, 1536)     [L, 1536]
  ├── + RoPE
  ├── 30× Vanilla DiT Block
  ├── Linear(1536, 32)     [L, 32]

输出: denoised_mat_latent  [L, 32]
  → SC-VAE Decoder → O-Voxel material → PBR纹理
```

---

### 2.3 DiT 核心差异总结

| 维度 | TRELLIS DiT | TRELLIS.2 DiT |
|------|------------|---------------|
| **输入Token数** | Stage1: 4096 (16³), Stage2: ~10K (打包后) | ~9.6K（直接处理，不打包） |
| **Latent维度** | 8 (结构) / 768 (SLat) | **32** (两个stage共用) |
| **网络深度** | 24层 (Large) | **30层** |
| **Hidden dim** | 1024 (Large) | **1536** |
| **MLP宽度** | 4× hidden | **8192**（≈5.3× hidden） |
| **参数量** | ~1.1B (L, 两阶段总计) | **~4B**（三阶段×1.3B/阶段） |
| **位置编码** | Sinusoidal | **RoPE**（旋转位置编码） |
| **Timestep调制** | adaLN | **adaLN-single**（更轻量） |
| **Condition来源** | CLIP(text) / DINOv2(image) | **DINOv3-L**（更强特征） |
| **Condition注入** | Cross-Attention | Cross-Attention（相同） |
| **Sparse Conv Packing** | ✅ 有（64³→32³缩短序列） | ❌ **去掉了** |
| **Skip Connection** | ✅ 有（U-Net style） | ❌ **去掉了** |
| **生成阶段** | 2阶段（结构+几何） | **3阶段**（结构+几何+材质） |

> **为什么TRELLIS.2能去掉packing和skip？**
> 因为SC-VAE的16×压缩已经把token数从~154K压到~9.6K、通道从768压到32了。Latent已经足够紧凑，DiT不需要再做额外的序列压缩技巧。这让DiT架构变得极其简洁——就是标准的Vanilla DiT。

---

## 三、整体Pipeline数据流对比

### TRELLIS 完整数据流

```
[3D Asset]
  → 渲染 150张多视图图像
  → DINOv2 提取2D特征
  → 投影聚合到 64³ 体素          [L≈20K, C_in]
  → SLat VAE Encoder (Transformer)  → SLAT [L≈20K, 768]  ← latent
  
生成时:
  G_S: noise[16³,8] → DiT → structure_latent[16³,8]
    → Decoder → binary_voxel[64³]
  G_L: noise[L,768] + positions → DiT(w/ packing+skip) → slat[L,768]
    → Decoder D_GS/D_RF/D_M → 3DGS / NeRF / Mesh
```

### TRELLIS.2 完整数据流

```
[3D Mesh]
  → O-Voxel转换（CPU，几秒）      shape[L≈154K, 7] + mat[L, 6]  @ 1024³
  → SC-VAE Encoder (稀疏卷积)
      shape → shape_latent [L'≈9.6K, 32]  @ 64³
      mat   → mat_latent   [L'≈9.6K, 32]  @ 64³

生成时:
  Stage1: noise → DiT → structure → active positions
  Stage2: noise[L',32] + positions → Vanilla DiT(30L,1536d) → shape_latent[L',32]
    → SC-VAE Decoder → O-Voxel shape → Mesh
  Stage3: noise[L',32] + shape_latent → Vanilla DiT → mat_latent[L',32]
    → SC-VAE Decoder → O-Voxel mat → PBR材质
```

---

## 四、关键设计哲学差异

1. **"2D→3D" vs "3D→3D"**
   - TRELLIS依赖多视图2D图像作为3D信息的中介（渲染→DINOv2→投影）
   - TRELLIS.2完全在原生3D空间操作（mesh→O-Voxel→稀疏卷积→latent）

2. **"Transformer VAE" vs "卷积VAE"**
   - TRELLIS用Transformer做VAE，好处是全局感受野，代价是序列长、通道宽(768)
   - TRELLIS.2用稀疏卷积做VAE，好处是高压缩率(16×)、latent紧凑(32通道)，代价是局部感受野（靠深度补偿——32个残差块）

3. **"复杂DiT" vs "简洁DiT"**
   - TRELLIS的DiT需要各种技巧（sparse conv packing、skip connection）来处理冗长的latent序列
   - TRELLIS.2因为latent已经足够紧凑，DiT可以回归最简形式（Vanilla DiT），反而把参数量堆到4B来提升质量

4. **"几何only" vs "几何+PBR"**
   - TRELLIS的纹理依赖后处理（多视图烘焙）
   - TRELLIS.2把材质生成做成了第三个端到端的DiT阶段，原生支持PBR（金属度、粗糙度、不透明度）

---

*基于 arXiv:2412.01506 (TRELLIS) 和 arXiv:2512.14692 (TRELLIS.2) 原文分析。*
