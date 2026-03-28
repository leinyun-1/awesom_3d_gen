# Native 3D Generation 发展脉络

> VAE + Diffusion 架构的原生3D生成技术演进

**整理日期**：2026年3月  
**核心思路**：在3D latent空间上训练扩散模型，实现端到端的原生3D生成。核心挑战在于设计高效的3D VAE和适配的Diffusion架构。

---

## 📑 论文列表（Quick Links）

### 核心主线论文

| 年份 | 论文 | 机构 | 会议/来源 | 关键词 |
|------|------|------|----------|-------|
| 2023.01 | [**3DShape2VecSet**: A 3D Shape Representation for Neural Fields and Generative Diffusion Models](https://arxiv.org/abs/2301.11445) | Adobe Research | SIGGRAPH 2023 | VecSet, Transformer VAE, 奠基 |
| 2023.06 | [**Michelangelo**: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation](https://arxiv.org/abs/2306.17115) | NVIDIA, 上海交大 | NeurIPS 2023 | 多模态对齐, SITA-VAE |
| 2024.03 | [**3DTopia**: Large Text-to-3D Generation Model with Hybrid Diffusion Priors](https://arxiv.org/abs/2403.02234) | 上海AI实验室, NTU | arXiv | PrimX, Hybrid Diffusion |
| 2024.05 | [**Direct3D**: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer](https://arxiv.org/abs/2405.14832) | 南京大学, DreamTech AI | NeurIPS 2024 | Triplane VAE, 可扩展 |
| 2024.06 | [**CLAY**: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets](https://arxiv.org/abs/2406.13897) | 智谱AI | arXiv | 第一个3D大模型, Flow Matching |
| 2024.12 | [**TRELLIS**: Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506) | Microsoft Research, 清华 | CVPR 2025 | 结构化latent (SLAT), 里程碑 |
| 2025.01 | [**Hunyuan3D 2.0**: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation](https://arxiv.org/abs/2501.12202) | 腾讯混元 | arXiv | 几何+纹理解耦, 商业标杆 |
| 2025.02 | [**TripoSG**: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models](https://arxiv.org/abs/2502.06608) | VAST AI (Tripo) | arXiv | MoE Transformer, 高保真 |
| 2025.05 | [**Direct3D-S2**: Gigascale 3D Generation Made Easy with Spatial Sparse Attention](https://arxiv.org/abs/2505.17412) | 南京大学, DreamTech AI | NeurIPS 2025 | 稀疏卷积VAE, 1024³ |
| 2025.06 | [**Hunyuan3D 2.1**: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material](https://arxiv.org/abs/2506.15442) | 腾讯混元 | arXiv | PBR材质, 完全开源 |
| 2025.06 | [**Hunyuan3D 2.5**: Towards High-Fidelity 3D Assets Generation with Ultimate Details](https://arxiv.org/abs/2506.16504) | 腾讯混元 | arXiv | LATTICE 10B, 768纹理 |
| 2025.09 | [**Hunyuan3D-Omni**: A Unified Framework for Controllable Generation of 3D Assets](https://arxiv.org/abs/2509.21245) | 腾讯混元 | arXiv | 统一多条件控制 |
| 2025.10 | [**Seed3D 1.0**: Simulation-Ready 3D Asset Generation from Single Images](https://arxiv.org/abs/2510.19944) | 字节跳动 Seed | arXiv | 仿真级, 6K纹理, PBR |
| 2025.12 | [**TRELLIS.2**: Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692) | Microsoft Research, 清华, USTC | arXiv | O-Voxel, 4B参数, 当前SOTA |

### 2025-2026 扩展论文

| 年份 | 论文 | 机构 | 关键词 |
|------|------|------|-------|
| 2024.08 | [**Atlas Gaussians Diffusion** for 3D Generation](https://arxiv.org/abs/2408.13055) | — | Atlas Gaussian表示, 局部patch |
| 2024.08 | [**OctFusion**: Octree-based Diffusion Models for 3D Shape Generation](https://arxiv.org/abs/2408.14732) | — | Octree扩散, 层次化3D |
| 2024.10 | [**L3DG**: Latent 3D Gaussian Diffusion](https://arxiv.org/abs/2410.13530) | — | 3DGS latent空间扩散 |
| 2024.11 | [**DiffusionGS**: Baking Gaussian Splatting into Diffusion Denoiser](https://arxiv.org/abs/2411.14384) | Adobe Research | 3DGS嵌入pixel diffusion |
| 2024.12 | [**Prometheus**: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation](https://arxiv.org/abs/2412.21117) | 浙江大学 | 场景级3D生成, GS-VAE |
| 2025.03 | [**COD-VAE**: Representing 3D Shapes With 64 Latent Vectors for 3D Diffusion Models](https://arxiv.org/abs/2503.08737) | — | 极致压缩 (仅64向量) |
| 2025.03 | [**SparseFlex** (TripoSF): High-Resolution and Arbitrary-Topology 3D Shape Modeling](https://arxiv.org/abs/2503.21732) | VAST AI (Tripo) | 任意拓扑, 高分辨率 |
| 2025.05 | [**Sparc3D**: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](https://arxiv.org/abs/2505.14521) | — | 稀疏表示, 1536³ |
| 2025.06 | [**PartCrafter**: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://arxiv.org/abs/2506.05573) | — | 部件级组合式生成 |
| 2025.09 | [**UniLat3D**: Geometry-Appearance Unified Latents for Single-Stage 3D Generation](https://arxiv.org/abs/2509.25079) | — | 几何-外观统一latent |
| 2025.12 | [**LATTICE**: Democratize High-Fidelity 3D Generation at Scale](https://arxiv.org/abs/2512.03052) | — | 大规模高保真生成 |
| 2025.12 | [**UltraShape 1.0**: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement](https://arxiv.org/abs/2512.21185) | 北京大学 | 渐进式几何细化 |
| 2025.12 | [**LaFiTe**: A Generative Latent Field for 3D Native Texturing](https://arxiv.org/abs/2512.04786) | — | 3D原生纹理生成 |

---

## 一、总体架构范式

Native 3D Gen 遵循 **VAE + Diffusion** 两阶段范式（类比2D LDM）：

1. **Stage 1 — 3D VAE**：将3D shape（mesh/点云/SDF体素等）编码到紧凑的latent空间，并从latent解码回3D表示。
2. **Stage 2 — Latent Diffusion**：在latent空间上训练扩散/流匹配模型，支持text/image/multi-view等条件引导。

> **核心竞争点**：VAE的设计是创新的激烈竞争点——一个高效的VAE意味着latent空间紧凑、重建几何质量高。  
> Diffusion侧则趋于同质化：基本都使用DiT架构，区别仅在于self-attention + cross-attention vs. mmDiT。

---

## 二、论文时间线 & 发展脉络

### Phase 1: 奠基期（2023）

#### 📄 3DShape2VecSet (SIGGRAPH 2023)
- **论文**：*3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models*
- **arXiv**：2301.11445 | **PDF**：`papers/01_3DShape2VecSet_2301.11445.pdf`
- **机构**：Adobe Research
- **核心贡献**：
  - **首次用Transformer实现3D shape的encode/decode**
  - 将点云/mesh编码为一组latent vectors（VecSet），解码为neural field
  - 使用cross-attention机制：query tokens从surface points上gather信息
  - 在latent vectors上训练DDPM扩散模型
- **VAE特点**：latent为1D token序列，**无位置信息**（unstructured latent）
- **局限**：latent缺乏空间结构，不利于空间编辑；数据量和分辨率有限
- **地位**：🏆 **奠基之作**，定义了"VecSet + Diffusion"的基本范式

#### 📄 Michelangelo (NeurIPS 2023)
- **论文**：*Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation*
- **arXiv**：2306.17115 | **PDF**：`papers/02_Michelangelo_2306.17115.pdf`
- **机构**：NVIDIA, 上海交通大学等
- **核心贡献**：
  - 提出 **"alignment-before-generation"** 策略
  - 构建Shape-Image-Text-Aligned VAE (SITA-VAE)，将3D shape、图像、文本对齐到共享latent空间
  - 在对齐的latent上训练条件DiT
- **VAE特点**：对齐多模态的latent表示，为后续multimodal condition奠定基础
- **地位**：多模态对齐思想的先驱

---

### Phase 2: 规模化探索（2024上半年）

#### 📄 CLAY (arXiv 2024.06)
- **论文**：*CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets*
- **arXiv**：2406.13897 | **PDF**：`papers/03_CLAY_2406.13897.pdf`
- **机构**：Zhipu AI (智谱)
- **核心贡献**：
  - **第一个3D生成大模型**（工程层面的里程碑）
  - 使用**多分辨率VAE** (Multi-resolution VAE) 提取丰富的3D先验
  - Diffusion使用 **Flow Matching**（而非DDPM）
  - 支持**多模态condition**：每个模态（text/image/voxel/point cloud）使用独立的cross-attention
  - **大规模数据**：显著扩大训练数据量
- **VAE特点**：多分辨率编码，层次化latent
- **地位**：🏆 **工程里程碑**，证明了原生3D生成可以scaling up

#### 📄 3DTopia (arXiv 2024.03) / 3DTopia-XL
- **论文**：*3DTopia: Large Text-to-3D Generation Model with Hybrid Diffusion Priors*
- **arXiv**：2403.02234 | **PDF**：`papers/10_3DTopia_2403.02234.pdf`
- **机构**：上海AI实验室 + 南洋理工
- **核心贡献**：
  - 两阶段T2-3D系统，使用hybrid diffusion prior
  - 3DTopia-XL引入 **PrimX**（primitive-based表示），将shape/texture/material编码为紧凑primitive
  - 支持导入标准图形学管线（游戏引擎、工业设计软件）
- **地位**：学术界的大规模3D生成探索

#### 📄 Direct3D (NeurIPS 2024)
- **论文**：*Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer*
- **arXiv**：2405.14832 | **PDF**：`papers/04_Direct3D_2405.14832.pdf`
- **机构**：南京大学（DreamTech AI）
- **核心贡献**：
  - **首个可扩展到wild input images的原生3D生成模型**
  - 提出 **D3D-VAE**：将triplane NeRF编码为token-based latent
  - 无需multiview diffusion或SDS优化
  - 在latent上训练 DiT with image condition (cross-attention)
- **VAE特点**：Triplane-based encoder
- **地位**：🏆 **学术界佼佼者**，简洁且高效

---

### Phase 3: 结构化latent革命（2024下半年）

#### 📄 TRELLIS (CVPR 2025, arXiv 2024.12)
- **论文**：*Structured 3D Latents for Scalable and Versatile 3D Generation*
- **arXiv**：2412.01506 | **PDF**：`papers/05_TRELLIS_2412.01506.pdf`
- **机构**：Microsoft Research + 清华大学
- **核心贡献**：
  - 🏆 **创造性提出 Structured LATent (SLAT)**
  - 将**特征体素 (feature voxels)** 作为结构化latent——latent自带3D位置信息
  - 从多视图编码→SLAT→解码为多种3D表示（mesh/3D Gaussian/radiance field）
  - 使用 **Rectified Flow** 训练扩散模型
  - **解码灵活性**：同一个latent可以decode为不同格式
- **VAE特点**：
  - Structured latent = 3D spatial位置 + 特征向量
  - 补足了3DShape2VecSet中latent缺乏位置信息的缺陷
  - 提高空间编辑能力
- **地位**：🏆 **结构化latent的里程碑**，深刻影响后续工作

---

### Phase 4: 商业落地与精度竞赛（2025-2026）

#### 📄 Hunyuan3D 2.0 (arXiv 2025.01)
- **论文**：*Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation*
- **arXiv**：2501.12202 | **PDF**：`papers/09_Hunyuan3D-2_2501.12202.pdf`
- **机构**：腾讯混元
- **核心贡献**：
  - 两大基础模型：**Hunyuan3D-DiT**（几何生成）+ **Hunyuan3D-Paint**（纹理生成）
  - 采用 **两阶段DiT架构**
  - 在纹理贴图上取得巨大进步，达到**精美级别**
  - 后续迭代 2.1 → 2.5 → 3.0，精度持续提升
- **地位**：🏆 **商业落地标杆**，腾讯入局的代表作

#### 📄 Hunyuan3D 2.1 (arXiv 2025.06)
- **论文**：*Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material*
- **arXiv**：2506.15442 | **PDF**：`papers/14_Hunyuan3D-2.1_2506.15442.pdf`
- **机构**：腾讯混元
- **GitHub**：[Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/tencent-hunyuan/hunyuan3d-2.1)
- **核心改进**：
  - 完全开源框架
  - **Physically-Based Rendering (PBR)** 纹理支持（Albedo + Metallic + Roughness）
  - Spatial-Aligned Multi-Attention、3D-Aware RoPE、Illumination-Invariant训练
  - 几何和纹理质量进一步提升

#### 📄 TripoSG (arXiv 2025.02)
- **论文**：*TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models*
- **arXiv**：2502.06608 | **PDF**：`papers/08_TripoSG_2502.06608.pdf`
- **机构**：VAST AI Research (Tripo)
- **核心贡献**：
  - 大规模 **Rectified Flow Model** 用于高保真3D shape合成
  - MoE Transformer架构
  - 高保真、高质量、高泛化
- **地位**：商业竞争者中的技术开源先锋

#### 📄 Direct3D-S2 (NeurIPS 2025, arXiv 2025.05)
- **论文**：*Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention*
- **arXiv**：2505.17412 | **PDF**：`papers/06_Direct3D-S2_2505.17412.pdf`
- **机构**：南京大学（DreamTech AI）
- **核心贡献**：
  - 🏆 **提出新VAE范式——Sparse SDF VAE (SS-VAE)**
  - 利用**稀疏卷积 + Transformer**作为encoder，高效处理高分辨率SDF
  - 实现 **高模输入、高模输出**（高达1024³分辨率）
  - **Spatial Sparse Attention (SSA)**：大幅降低计算开销
  - 仅需8 GPU即可训练1024³分辨率模型
- **VAE特点**：
  - 稀疏SDF体积→稀疏卷积编码→稀疏latent表示
  - 保留空间结构的同时极大压缩计算量
- **地位**：🏆 **VAE创新的重要里程碑**，证明了稀疏卷积在3D VAE中的巨大潜力

#### 📄 TRELLIS.2 (arXiv 2025.12)
- **论文**：*Native and Compact Structured Latents for 3D Generation*
- **arXiv**：2512.14692 | **PDF**：`papers/07_TRELLIS2_2512.14692.pdf`
- **机构**：Microsoft Research + 清华大学 + USTC
- **核心贡献**：
  - **4B参数**的大规模image-to-3D模型
  - 提出 **O-Voxel**：一种"field-free"稀疏体素表示
  - 从原生3D数据学习VAE（不再依赖多视图编码）
  - **16倍空间压缩**
  - 支持最高 **1536³** 的PBR纹理资产生成
  - 三阶段管线：几何生成→纹理生成→PBR纹理增强
- **VAE特点**：
  - O-Voxel：Instant Bidirectional Conversion（mesh ↔ O-Voxel即时双向转换）
  - Sparse Compression VAE编码
- **地位**：🏆 **当前SOTA**，代表了native 3D gen的最新技术前沿

---

## 三、VAE架构演进对比

| 论文 | 时间 | 输入 | Latent类型 | Encoder | 空间结构 | 分辨率 |
|------|------|------|-----------|---------|---------|-------|
| **3DShape2VecSet** | 2023.01 | 点云/mesh | 1D token序列 | Transformer (cross-attn) | ❌ 无位置信息 | 低 |
| **Michelangelo** | 2023.06 | 点云 | 对齐latent | Shape-Image-Text对齐 | ❌ | 低 |
| **CLAY** | 2024.06 | 多种 | 多分辨率latent | Multi-res VAE | 部分 | 中 |
| **Direct3D** | 2024.05 | image | Triplane tokens | Triplane encoder | 部分 | 中 |
| **TRELLIS** | 2024.12 | 多视图 | **结构化体素latent (SLAT)** | 多视图编码 | ✅ 3D位置 | 高 |
| **Direct3D-S2** | 2025.05 | image/SDF | **稀疏SDF latent** | 稀疏卷积+Transformer | ✅ 稀疏空间 | **1024³** |
| **TRELLIS.2** | 2025.12 | image/mesh | **O-Voxel latent** | Sparse Compression VAE | ✅ 原生3D | **1536³** |

---

## 四、Diffusion架构演进

| 论文 | Diffusion方法 | 架构 | Condition方式 |
|------|-------------|------|-------------|
| **3DShape2VecSet** | DDPM | U-Net style | 无条件 |
| **Michelangelo** | DDPM | DiT | Cross-attention (text/image) |
| **CLAY** | **Flow Matching** | DiT | 每个模态独立cross-attention |
| **TRELLIS** | **Rectified Flow** | DiT | Cross-attention |
| **Direct3D** | DDPM/EDM | DiT | Image cross-attention |
| **TripoSG** | **Rectified Flow** | MoE Transformer | Image condition |
| **Direct3D-S2** | Flow Matching | DiT with SSA | Image cross-attention |
| **Hunyuan3D 2** | Flow Matching | DiT | Multi-modal |
| **TRELLIS.2** | **Rectified Flow** | DiT | Image condition |

> **趋势**：Diffusion训练方法从DDPM → EDM → Flow Matching/Rectified Flow 演进，后者训练更稳定、采样更高效。

---

## 五、关键技术演进脉络总结

```
3DShape2VecSet (2023.01)
│  开创 VecSet + Transformer + Diffusion 范式
│  局限：latent无位置信息，数据量小
│
├─ Michelangelo (2023.06)
│    多模态对齐的latent空间
│
├─ CLAY (2024.06)
│    规模化：大数据 + Flow Matching + 多模态condition
│    第一个3D生成大模型
│
├─ Direct3D (2024.05)
│    Triplane-based VAE，首个可扩展wild image输入
│
├─ 3DTopia / 3DTopia-XL (2024.03)
│    Primitive-based表示，学术探索
│
└─ TRELLIS (2024.12)  ← 关键转折点
|     结构化latent (SLAT)：特征体素 = 位置 + 特征
|     补足了位置信息缺失
│     支持多种3D表示输出
│
├─ Hunyuan3D 2.0/2.1/2.5/3.0 (2025.01-2026)
│    商业落地，纹理质量精美
│
├─ TripoSG (2025.02)
│    Rectified Flow + MoE，商业高保真
│
├─ Direct3D-S2 (2025.05)
│    稀疏卷积VAE，高模输入高模输出
│    1024³分辨率突破
│
└─ TRELLIS.2 (2025.12)
    O-Voxel + 原生3D VAE，4B参数
    当前SOTA，1536³ PBR
```

---

## 六、本地论文PDF清单

`papers/` 文件夹中收录了以下论文的PDF（共27篇）：

| # | 文件名 | 论文 | arXiv链接 |
|---|--------|------|----------|
| 01 | `01_3DShape2VecSet_2301.11445.pdf` | 3DShape2VecSet | [2301.11445](https://arxiv.org/abs/2301.11445) |
| 02 | `02_Michelangelo_2306.17115.pdf` | Michelangelo | [2306.17115](https://arxiv.org/abs/2306.17115) |
| 03 | `03_CLAY_2406.13897.pdf` | CLAY | [2406.13897](https://arxiv.org/abs/2406.13897) |
| 04 | `04_Direct3D_2405.14832.pdf` | Direct3D | [2405.14832](https://arxiv.org/abs/2405.14832) |
| 05 | `05_TRELLIS_2412.01506.pdf` | TRELLIS | [2412.01506](https://arxiv.org/abs/2412.01506) |
| 06 | `06_Direct3D-S2_2505.17412.pdf` | Direct3D-S2 | [2505.17412](https://arxiv.org/abs/2505.17412) |
| 07 | `07_TRELLIS2_2512.14692.pdf` | TRELLIS.2 | [2512.14692](https://arxiv.org/abs/2512.14692) |
| 08 | `08_TripoSG_2502.06608.pdf` | TripoSG | [2502.06608](https://arxiv.org/abs/2502.06608) |
| 09 | `09_Hunyuan3D-2_2501.12202.pdf` | Hunyuan3D 2.0 | [2501.12202](https://arxiv.org/abs/2501.12202) |
| 10 | `10_3DTopia_2403.02234.pdf` | 3DTopia | [2403.02234](https://arxiv.org/abs/2403.02234) |
| 11 | `11_COD-VAE_2503.08737.pdf` | COD-VAE | [2503.08737](https://arxiv.org/abs/2503.08737) |
| 12 | `12_AtlasGaussians_2408.13055.pdf` | Atlas Gaussians | [2408.13055](https://arxiv.org/abs/2408.13055) |
| 13 | `13_L3DG_2410.13530.pdf` | L3DG | [2410.13530](https://arxiv.org/abs/2410.13530) |
| 14 | `14_Hunyuan3D-2.1_2506.15442.pdf` | Hunyuan3D 2.1 | [2506.15442](https://arxiv.org/abs/2506.15442) |
| 15 | `15_OctFusion_2408.14732.pdf` | OctFusion | [2408.14732](https://arxiv.org/abs/2408.14732) |
| 16 | `16_LATTICE_2512.03052.pdf` | LATTICE | [2512.03052](https://arxiv.org/abs/2512.03052) |
| 17 | `17_UniLat3D_2509.25079.pdf` | UniLat3D | [2509.25079](https://arxiv.org/abs/2509.25079) |
| 18 | `18_UltraShape_2512.21185.pdf` | UltraShape 1.0 | [2512.21185](https://arxiv.org/abs/2512.21185) |
| 19 | `19_Sparc3D_2505.14521.pdf` | Sparc3D | [2505.14521](https://arxiv.org/abs/2505.14521) |
| 20 | `20_SparseFlex_2503.21732.pdf` | SparseFlex (TripoSF) | [2503.21732](https://arxiv.org/abs/2503.21732) |
| 21 | `21_LaFiTe_2512.04786.pdf` | LaFiTe | [2512.04786](https://arxiv.org/abs/2512.04786) |
| 23 | `23_Prometheus_2412.21117.pdf` | Prometheus | [2412.21117](https://arxiv.org/abs/2412.21117) |
| 24 | `24_DiffusionGS_2411.14384.pdf` | DiffusionGS | [2411.14384](https://arxiv.org/abs/2411.14384) |
| 25 | `25_PartCrafter_2506.05573.pdf` | PartCrafter | [2506.05573](https://arxiv.org/abs/2506.05573) |
| 27 | `27_Hunyuan3D-2.5_2506.16504.pdf` | Hunyuan3D 2.5 | [2506.16504](https://arxiv.org/abs/2506.16504) |
| 28 | `28_Hunyuan3D-Omni_2509.21245.pdf` | Hunyuan3D-Omni | [2509.21245](https://arxiv.org/abs/2509.21245) |
| 29 | `29_Seed3D_2510.19944.pdf` | Seed3D 1.0 | [2510.19944](https://arxiv.org/abs/2510.19944) |

---

## 七、评论性文章参考

详见 `comments/` 文件夹：

- `01_TRELLIS技术解读_知乎.md` — 解读TRELLIS的结构化latent设计
- `02_TRELLIS2_vs_Hunyuan3D_对比.md` — 微软与腾讯技术交锋分析
- `03_Direct3D_论文分享_知乎.md` — Direct3D的D3D-VAE和triplane设计解读
- `04_CLAY_论文阅读_知乎.md` — CLAY多分辨率VAE和flow matching分析

---

*本文档将持续更新。如有新的重要论文发表，请补充至对应Phase。*
