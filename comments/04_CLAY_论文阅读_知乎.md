# CLAY 论文阅读：第一个3D生成大模型

> 来源：知乎
> 原文链接：https://zhuanlan.zhihu.com/p/710458934

## 论文背景

CLAY 由智谱AI提出，是**工程意义上的第一个3D生成大模型**。其核心架构由两部分组成：
1. **Multi-resolution VAE**：多分辨率的3D变分自编码器
2. **Minimalistic Latent DiT**：极简主义的latent扩散Transformer

## 核心技术解析

### 1. 多分辨率 VAE

CLAY 使用多分辨率方案提取3D先验：
- **低分辨率层**：捕捉全局几何结构
- **高分辨率层**：捕捉局部细节
- 多层latent通过层次化方式组合

### 2. Flow Matching 替代 DDPM

CLAY 是3D生成领域较早采用 **Flow Matching** 的工作：
- 训练更稳定（相比DDPM的噪声调度敏感性）
- 采样更高效
- 理论基础更简洁（ODE路径而非SDE）

### 3. 多模态 Condition

CLAY 的一大亮点是支持多模态条件输入，设计巧妙：
- **每个模态使用独立的cross-attention层**
- 支持 text / image / voxel / point cloud 等多种输入
- 不同模态的信息通过各自的cross-attention注入latent

### 4. 大规模数据

CLAY 显著扩大了训练数据量，这是其能scaling up的关键因素之一。大数据量确保了：
- 更丰富的几何先验
- 更好的泛化能力
- 对多种输入的鲁棒处理

## 产品化

CLAY 已产品化为 **Rodin Gen-1**，用于高质量3D资产创建。

## 影响

CLAY 的意义更多在于工程层面的里程碑：
1. **证明了Native 3D Gen可以scaling up**
2. Flow Matching 和多模态cross-attention成为后续工作的标准配置
3. 多分辨率VAE思想影响了后续的层次化3D表示研究

## 局限

- VAE的latent仍然不够"结构化"（TRELLIS后来解决了这个问题）
- 生成精度在后续工作（Hunyuan3D、TRELLIS.2）中被超越
