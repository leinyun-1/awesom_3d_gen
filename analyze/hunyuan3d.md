# Hunyuan3D 系列演进分析：2.0 → 2.1 → 2.5

> 以数据流视角分析腾讯混元3D系列在形状生成和纹理生成两条管线上的关键组件升级。
> 
> **论文来源**：
> - Hunyuan3D 2.0：arXiv:2501.12202（2025.01）
> - Hunyuan3D 2.1：arXiv:2506.15442（2025.06）
> - Hunyuan3D 2.5：arXiv:2506.16504（2025.06）
> - Hunyuan3D 3.0：尚未发表正式论文（截至2026.03）

---

## 〇、整体架构：两条管线始终不变

混元3D系列从2.0到2.5始终保持**形状-纹理解耦的两阶段管线**：

```
Stage 1: Image → Hunyuan3D-DiT → Shape Latent → ShapeVAE Decoder → 3D Mesh（白模）
Stage 2: 3D Mesh + Reference Image → Hunyuan3D-Paint → 多视角纹理图 → UV烘焙 → 纹理贴图
```

每一代的升级都是在这两条管线内部做组件替换和增强，而非重构架构范式。

---

## 一、形状生成管线（Stage 1）的演进

### 1.1 ShapeVAE：从2.0到2.1基本不变，2.5整体替换

#### Hunyuan3D 2.0 / 2.1 的 ShapeVAE

```
【Encoder 数据流】

输入: 3D mesh 表面
  ├── 采样点云: uniform_points P_u + importance_points P_i
  │     P_u: 均匀采样                    [N_u, 6]    (xyz + normal)
  │     P_i: 重要性采样（边缘/角落加密）    [N_i, 6]    (xyz + normal)
  │
  ├── Fourier特征编码                    [N, D_fourier]
  ├── FPS选取查询点 Q_u, Q_i
  ├── Cross-Attention: Q从P中聚合特征
  │     query = Q_u ∪ Q_i               [M, D]       M ≤ 3072
  │     key/value = Fourier(P)
  │
  ├── 预测 μ, logσ²                     [M, d_0] × 2
  ├── 重参数化采样
  │
输出: shape_latent_tokens               [M, d_0]     M可变，最长3072

【Decoder 数据流】

输入: shape_latent_tokens               [M, d_0]
      query_grid_points Q_g             [N_grid, 3]   3D网格采样点
  │
  ├── Point Perceiver: Q_g从latent中提取特征
  │     query = Q_g                     [N_grid, 3]
  │     key/value = shape_latent        [M, d_0]
  │
  ├── 预测 SDF 值                       [N_grid, 1]
  ├── Marching Cubes 提取等值面
  │
输出: triangle_mesh
```

**关键参数**：
- 序列长度 M：可变，最长 **3072** tokens
- SDF采样点数：N_near=249,856 + N_uniform=249,856 ≈ **50万**
- 表面采样点数：N=124,928（Random + Sharp各半）

**2.1相比2.0的ShapeVAE改进**：
- 改进不大，主要是 **importance sampling策略的细化**——在网格边缘和角落处采样更多点以捕获高频细节
- 变长token支持从一开始就有（2.0已实现），2.1对其训练策略做了优化

#### Hunyuan3D 2.5 的 ShapeVAE：整体替换为 LATTICE

2.5 将形状基础模型完全替换为 **LATTICE**（独立论文 arXiv:2512.03052），这是一个**全新的形状生成系统**，不再使用2.0/2.1的VecSet-based ShapeVAE。

```
LATTICE 形状系统（2.5）:
  - 参数规模: 最大 10B（100亿），远超2.0/2.1的DiT参数量
  - VAE: 基于稀疏卷积的3D VAE（与TRELLIS.2的SC-VAE思路类似）
  - 生成模型: 大规模DiT
  - 输出: 高精度mesh（干净拓扑+平滑表面+锐利边缘）
```

> **这是2.0→2.5最大的断裂式升级**：ShapeVAE从"点云→Cross-Attention→VecSet latent→SDF"范式，切换为"稀疏体素→稀疏卷积→结构化latent→直出mesh"范式。

---

### 1.2 Shape DiT：条件注入和架构的持续升级

#### Hunyuan3D 2.0 的 DiT

```
输入:
  noise_tokens x_t            [N, d_0]       N ≤ 3072, d_0为latent维度
  condition_tokens c           [P, D_dino]    DINOv2 Giant图像patch特征
  timestep t                   [1]

架构: FLUX风格 双流→单流 DiT
  │
  ├── 【双流模块】× K层
  │     latent_stream:  Linear → QKV → Attention(与cond交互) → MLP
  │     cond_stream:    Linear → QKV → Attention(与latent交互) → MLP
  │     两个stream各自独立投影，Attention时拼接做joint attention
  │
  ├── 【单流模块】× L层
  │     concat(latent, cond) → [N+P, D]
  │     并行: Spatial-Attention + Channel-Attention → MLP
  │
  ├── 取出latent部分
  │
输出: predicted_velocity u_θ   [N, d_0]

关键设计决策:
  - 图像编码器: DINOv2 Giant，输入 518×518
  - ⚠️ 省略了latent序列的位置编码！
    理由: 3D latent token的内容本身决定其在3D空间的位置
         （不像2D图像token有固定的空间像素位置）
  - 训练目标: Flow Matching (Rectified Flow)
  - ODE求解: 一阶 Euler
```

#### Hunyuan3D 2.1 的 DiT

相比2.0的变化很小：
- **层数**: 21层 Transformer
- **图像编码器**: 仍然是 DINOv2 Giant, 518×518
- **训练目标**: 同样是 Flow Matching
- **主要改进在训练策略而非架构**

#### Hunyuan3D 2.5 的 DiT

随着ShapeVAE替换为LATTICE，DiT也被整体替换：
- **参数量跃升**：从~1B量级 → **10B**
- **引导生成 + 步骤蒸馏**：生成效率提升3倍
- 具体架构细节尚未在论文中完整公开

---

## 二、纹理生成管线（Stage 2）的演进

纹理管线是混元3D系列升级最频繁、改动最大的部分。

### 2.1 Hunyuan3D-Paint 2.0：基线系统

```
输入:
  reference_image              [1, 3, 512, 512]     参考图像
  canonical_normal_maps        [V, 3, 512, 512]     V个视角的法线图
  canonical_coord_maps (CCM)   [V, 3, 512, 512]     V个视角的坐标图
  noise                        [V, 4, 64, 64]       V个视角的latent噪声

预处理:
  ├── SD2.1 VAE Encoder:
  │     normal_maps → geo_latent          [V, 4, 64, 64]
  │     coord_maps  → coord_latent        [V, 4, 64, 64]
  ├── concat(noise, geo_latent, coord_latent)
  │                                       [V, 12, 64, 64]   ← 通道拼接
  │
  ├── Reference-Net（冻结的SD2.1权重，t=0）:
  │     reference_image → ref_features     多层特征

UNet处理:
  对每个视角的 [12, 64, 64] 输入:
  ├── Conv (12→320 通道)
  ├── 标准SD2.1 UNet骨干
  │     每个Self-Attention层替换为:
  │     Z_out = Z_SA
  │           + λ_ref · RefAttn(Q, K_ref, V_ref)      ← 参考图像特征注入
  │           + λ_mv  · MVAttn(Q, K_other_views, V_other_views)  ← 多视角一致性
  │
  ├── 输出 latent                         [V, 4, 64, 64]
  ├── SD2.1 VAE Decoder → RGB图像         [V, 3, 512, 512]

后处理:
  ├── ESRGAN 超分辨率（可选）
  ├── 多视角 UV 烘焙 → 纹理贴图

关键参数:
  - 基础模型: SD2.1 ZSNR checkpoint
  - 视角数量: 训练6个/推理8-12个（从44个预设视角池选取）
  - 分辨率: 512×512
  - 输出: RGB纹理（非PBR）
```

### 2.2 Hunyuan3D-Paint 2.1：三大关键升级

2.1在2.0的基础上做了三处重要改进，纹理质量和材质能力显著提升：

#### 升级①：PBR多通道输出

```
2.0: 输出 RGB 纹理           [V, 3, 512, 512]
2.1: 输出 Albedo + MR 纹理    Albedo [V, 3, 512, 512] + Metallic-Roughness [V, 2, 512, 512]

Albedo（反照率）: 光照无关的材质颜色
Metallic（金属度）: 0~1
Roughness（粗糙度）: 0~1
→ 符合 Disney Principled BRDF
→ 支持在任意光照环境下PBR渲染
```

#### 升级②：Spatial-Aligned Multi-Attention Module

```
问题: Albedo分支和MR分支独立生成时，空间位置容易不对齐

2.0的注意力:
  Z_SA → + RefAttn → + MVAttn → 输出

2.1新增的空间对齐:
  Albedo分支:  Z_SA → + RefAttn_albedo → + MVAttn → albedo_features
  MR分支:      Z_SA → + RefAttn_MR    → + MVAttn → MR_features
                         ↑
                    直接接收 albedo RefAttn 的输出
                    （albedo分支的参考注意力输出传播到MR分支）

效果: 金属度/粗糙度的空间分布与反照率严格对齐
```

#### 升级③：3D-Aware RoPE

```
问题: 跨视角的纹理接缝和重影伪影

2.0: 使用可学习的相机ID嵌入（离散）
2.1: 引入 3D-Aware RoPE
  ├── 将多分辨率3D坐标编码（来自CCM）
  ├── 与UNet各层hidden state做加性融合
  ├── 使RoPE位置编码携带3D空间信息
  
效果: 跨视角一致性显著提升，减少接缝
```

#### 升级④：Illumination-Invariant Training

```
问题: 训练数据中的光照baked进纹理，导致Albedo不纯净

2.0: 均匀白光渲染 + 去光照模块（后处理）
2.1: 训练时主动引入光照不变性约束
  ├── 同一物体在不同光照条件下渲染
  │     p=0.3 使用点光源
  │     p=0.7 使用HDR环境贴图
  ├── 计算 illumination-invariant consistency loss
  │     强制模型输出的Albedo在不同光照下保持一致
  
效果: Albedo更纯净，光照与材质解耦更彻底
```

**2.1训练参数**：
- 优化器: AdamW, lr=5e-5
- 训练步数: 80,000
- Batch size: 48
- Warm-up: 2,000步
- 训练成本: ~180 GPU-days
- 视角: 每个仰角采样24个方位角, 仰角 {-20°, 0°, 20°, random}

### 2.3 Hunyuan3D-Paint 2.5：分辨率+共享注意力

2.5在2.1基础上的主要纹理改进：

#### 升级①：渐进式分辨率提升

```
2.0/2.1: 全程 512×512 → (可选ESRGAN超分)

2.5: 两阶段渐进式
  Stage A: 512×512 — 确立整体材质分布
  Stage B: 768×768 — "显微级"精细纹理生成

最终输出: 768×768 纹理贴图（原生，非超分）
```

#### 升级②：共享注意力机制（Shared Attention）

```
2.1: Albedo RefAttn输出 → 传播到MR分支（单向传播）

2.5: 三通道（Albedo, Roughness, Metallic）动态协同
  ├── 共享注意力：三个通道在注意力层实时通讯
  ├── 类比"三位化妆师实时对讲"
  
效果: 材质通道间的空间一致性进一步提升
```

#### 升级③：UniPC采样加速

```
2.0/2.1: 标准DDIM采样（步数较多）
2.5: UniPC采样器 — 更少步数达到同等质量
```

---

## 三、各版本关键组件演进总表

### 3.1 Shape管线

| 组件 | 2.0 | 2.1 | 2.5 |
|------|-----|-----|-----|
| **ShapeVAE类型** | VecSet (Cross-Attn) | VecSet (改进采样) | ⚡**LATTICE** (稀疏卷积) |
| **3D表示** | 点云→SDF→Marching Cubes | 同上 | 稀疏体素→直出mesh |
| **Latent形式** | 1D token序列 [M≤3072, d₀] | 同上 | 结构化稀疏latent |
| **位置信息** | ❌ 无位置编码 | ❌ | ✅ 稀疏卷积隐式保留 |
| **DiT架构** | FLUX双流→单流 | 21层Transformer | 10B参数DiT |
| **图像编码器** | DINOv2 Giant | DINOv2 Giant | 未公开 |
| **DiT Condition** | Joint Attention | 同上 | 未公开 |
| **训练目标** | Flow Matching | Flow Matching | Flow Matching + 步骤蒸馏 |

### 3.2 Texture管线

| 组件 | 2.0 | 2.1 | 2.5 |
|------|-----|-----|-----|
| **基础模型** | SD2.1 ZSNR | SD2.1 ZSNR | SD2.1 ZSNR (扩展) |
| **输出通道** | RGB (3ch) | ⚡**Albedo + MR** (5ch) | Albedo + MR (5ch) |
| **PBR支持** | ❌ | ✅ Disney BRDF | ✅ 进一步增强 |
| **分辨率** | 512×512 | 512×512 | ⚡**768×768** (渐进式) |
| **参考注意力** | RefAttn (冻结SD2.1) | RefAttn + 空间对齐传播 | ⚡**共享注意力** (三通道联动) |
| **多视角注意力** | MVAttn | MVAttn | MVAttn (增强) |
| **位置编码** | 可学习相机ID嵌入 | ⚡**3D-Aware RoPE** | 3D-Aware RoPE |
| **光照处理** | 白光渲染+去光照后处理 | ⚡**Illumination-Invariant训练** | 同上 (强化) |
| **几何条件** | Normal + CCM (通道拼接) | Normal + CCM | Normal + CCM |
| **采样器** | DDIM | DDIM | ⚡**UniPC** |
| **超分** | ESRGAN (后处理) | ESRGAN (后处理) | 原生768（无需超分） |

### 3.3 系统级

| 指标 | 2.0 | 2.1 | 2.5 |
|------|-----|-----|-----|
| **总参数量** | ~1B | ~1B | ⚡**~10B** |
| **训练数据** | 大规模3D数据集 | 同上+更多光照变体 | 扩展高质量数据集 |
| **开源程度** | 部分开源 | ⚡完全开源+PBR | 完全开源 |
| **生成速度** | 基线 | 基线 | ⚡**3倍加速** (步骤蒸馏) |

---

## 四、关键洞察

### 1. Shape管线的"范式跳跃"发生在2.5

2.0→2.1是连续优化（更好的采样策略、训练微调），但2.5是**不连续的范式替换**：
- 从 **VecSet（1D无结构latent + SDF）** 切换到 **LATTICE（结构化稀疏latent + 直出mesh）**
- 这与整个native 3D gen领域的趋势一致：从无结构latent向结构化latent演进

### 2. Texture管线是持续渐进式升级

每一代都在相同的SD2.1基础上做增量改进：
- 2.0建立了多注意力（Ref + MV）基线
- 2.1加入了**PBR + 空间对齐 + 3D RoPE + 光照不变**四项关键技术
- 2.5进一步做了**分辨率提升 + 共享注意力 + 采样加速**

2.1是纹理管线改动最大的版本。

### 3. 2.0的"无位置编码"设计值得注意

Hunyuan3D-DiT选择**不对latent序列加位置编码**的设计决策很独特。理由是3D latent token的内容（SDF值分布）本身就隐含了空间位置信息——这跟2D图像不同（2D patch的位置必须由PE告知模型）。这个设计在TRELLIS/TRELLIS.2中是反过来的（它们显式使用3D位置编码或RoPE）。

### 4. Hunyuan3D 3.0 尚无论文

截至2026年3月，Hunyuan3D 3.0尚未发表正式论文。从产品端的信息看，3.0主要在生成精度和纹理精美度上进一步提升，但具体的架构改动细节未公开。

---

*基于 arXiv:2501.12202 (2.0), arXiv:2506.15442 (2.1), arXiv:2506.16504 (2.5) 原文分析。*
