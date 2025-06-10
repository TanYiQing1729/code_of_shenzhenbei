# 基于基因型数据的DNA混合物比例估算

## 1. 数学建模基础

### 1.1 问题的数学描述

考虑包含 $n$ 个贡献者的DNA混合样本，在STR基因座上的分析问题：

$$
\mathcal{P} = \{P_1, P_2, \ldots, P_n\} \quad \text{(贡献者集合)}
$$

每个贡献者 $P_j$ 的混合比例为 $r_j$，满足约束：

$$
\sum_{j=1}^{n} r_j = 1, \quad r_j \geq 0, \quad \forall j \in \{1,2,\ldots,n\}
$$

### 1.2 基因型数学表示

对于贡献者 $P_j$ 在基因座的基因型：

$$
G_j = (a_{j1}, a_{j2}) \quad \text{其中 } a_{ji} \in \mathcal{A}
$$

定义等位基因贡献函数：

$$
\delta_{ij} = \begin{cases}
2 & \text{if } a_{j1} = a_{j2} = a_i \text{ (纯合子)} \\
1 & \text{if } |\{a_{j1}, a_{j2}\} \cap \{a_i\}| = 1 \text{ (杂合子)} \\
0 & \text{otherwise}
\end{cases}
$$

### 1.3 峰高生成的数学模型

基于Hardy-Weinberg平衡和等位基因剂量效应：

$$
\mathbb{E}[h_i] = \alpha \sum_{j=1}^{n} r_j \cdot \delta_{ij}
$$

其中：

- $h_i$：等位基因 $a_i$ 的观测峰高
- $\alpha > 0$：基因座特异性放大系数
- $\delta_{ij}$：贡献者 $j$ 对等位基因 $a_i$ 的贡献

## 2. 线性方程组的建立

### 2.1 基因型矩阵构造

定义基因型矩阵 $\mathbf{G} \in \mathbb{R}^{m \times n}$：

$$
\mathbf{G} = \begin{pmatrix}
\delta_{11} & \delta_{12} & \cdots & \delta_{1n} \\
\delta_{21} & \delta_{22} & \cdots & \delta_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\delta_{m1} & \delta_{m2} & \cdots & \delta_{mn}
\end{pmatrix}
$$

其中 $m$ 是观测到的等位基因数量。

### 2.2 线性系统的数学表达

忽略常数项 $\alpha$（关注相对比例），得到核心线性方程组：

$$
\mathbf{G} \mathbf{r} = \mathbf{h}
$$

展开形式：

$$
\begin{pmatrix}
\delta_{11} & \delta_{12} & \cdots & \delta_{1n} \\
\delta_{21} & \delta_{22} & \cdots & \delta_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\delta_{m1} & \delta_{m2} & \cdots & \delta_{mn}
\end{pmatrix}
\begin{pmatrix}
r_1 \\ r_2 \\ \vdots \\ r_n
\end{pmatrix}
=
\begin{pmatrix}
h_1 \\ h_2 \\ \vdots \\ h_m
\end{pmatrix}
$$

### 2.3 约束优化问题

完整的数学问题表述：

$$
\begin{align}
\min_{\mathbf{r}} \quad & \frac{1}{2} \|\mathbf{G} \mathbf{r} - \mathbf{h}\|_2^2 \tag{P} \\
\text{s.t.} \quad & \mathbf{1}^T \mathbf{r} = 1 \\
& \mathbf{r} \geq \mathbf{0}
\end{align}
$$

## 3. 遇到的关键数学问题

### 3.1 问题1：系统不相容性

**数学描述**：
观测数据含噪声，精确解不存在：

$$
\mathbf{G} \mathbf{r} = \mathbf{h} + \boldsymbol{\epsilon}
$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$

**数学解决方案**：
采用最小二乘准则：

$$
\mathbf{r}^* = \arg\min_{\mathbf{r}} \|\mathbf{G} \mathbf{r} - \mathbf{h}\|_2^2
$$

**理论依据**：
在高斯噪声假设下，最小二乘估计是最大似然估计：

$$
\mathcal{L}(\mathbf{r}) = -\frac{1}{2\sigma^2} \|\mathbf{G} \mathbf{r} - \mathbf{h}\|_2^2 + \text{const}
$$

### 3.2 问题2：矩阵条件数问题

**数学描述**：
当贡献者基因型相似时，$\mathbf{G}$ 接近奇异：

$$
\kappa(\mathbf{G}) = \frac{\sigma_{\max}(\mathbf{G})}{\sigma_{\min}(\mathbf{G})} \gg 1
$$

**误差放大效应**：

$$
\frac{\|\Delta \mathbf{r}\|_2}{\|\mathbf{r}\|_2} \leq \kappa(\mathbf{G}) \frac{\|\Delta \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
$$

**数学解决方案**：
正则化方法 - Tikhonov正则化：

$$
\mathbf{r}^{(ridge)} = \arg\min_{\mathbf{r}} \|\mathbf{G} \mathbf{r} - \mathbf{h}\|_2^2 + \alpha \|\mathbf{r}\|_2^2
$$

解析解：

$$
\mathbf{r}^{(ridge)} = (\mathbf{G}^T \mathbf{G} + \alpha \mathbf{I})^{-1} \mathbf{G}^T \mathbf{h}
$$

**正则化参数选择**：

$$
\alpha = \beta \cdot \frac{\text{tr}(\mathbf{G}^T \mathbf{G})}{n}
$$

其中 $\beta = 0.01$ 是经验参数。

### 3.3 问题3：奇异矩阵处理

**数学描述**：
当 $\text{rank}(\mathbf{G}) < \min(m, n)$ 时，标准方法失效。

**数学解决方案**：
Moore-Penrose伪逆：

$$
\mathbf{r}^{(pinv)} = \mathbf{G}^+ \mathbf{h}
$$

**SVD计算**：

$$
\mathbf{G} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
$$

$$
\mathbf{G}^+ = \mathbf{V} \boldsymbol{\Sigma}^+ \mathbf{U}^T
$$

其中：

$$
[\boldsymbol{\Sigma}^+]_{ii} = \begin{cases}
\sigma_i^{-1} & \text{if } \sigma_i > \tau \\
0 & \text{if } \sigma_i \leq \tau
\end{cases}
$$

数值截断阈值：

$$
\tau = \max(m, n) \cdot \|\mathbf{G}\|_2 \cdot \epsilon_{\text{machine}}
$$

## 4. 核心求解算法的数学原理

### 4.1 非负最小二乘 (NNLS) 算法

**数学问题**：

$$
\min_{\mathbf{r} \geq \mathbf{0}} \|\mathbf{G} \mathbf{r} - \mathbf{h}\|_2^2
$$

**KKT条件**：

$$
\begin{align}
\mathbf{G}^T (\mathbf{G} \mathbf{r}^* - \mathbf{h}) &= \boldsymbol{\lambda} \\
\mathbf{r}^* &\geq \mathbf{0} \\
\boldsymbol{\lambda} &\geq \mathbf{0} \\
\boldsymbol{\lambda}^T \mathbf{r}^* &= 0
\end{align}
$$

**Lawson-Hanson算法核心**：
设 $\mathcal{P} = \{j : r_j > 0\}$（正集），$\mathcal{Z} = \{j : r_j = 0\}$（零集）

迭代公式：

$$
\mathbf{r}_{\mathcal{P}}^{(k+1)} = (\mathbf{G}_{\mathcal{P}}^T \mathbf{G}_{\mathcal{P}})^{-1} \mathbf{G}_{\mathcal{P}}^T \mathbf{h}
$$

**收敛性定理**：
Lawson-Hanson算法在有限步内收敛到全局最优解，步数上界：

$$
k_{\max} \leq 3n
$$

### 4.2 多方法融合的数学理论

**中位数估计器**：
对于估计向量集合 $\{\mathbf{r}^{(1)}, \mathbf{r}^{(2)}, \mathbf{r}^{(3)}\}$：

$$
r_j^{(final)} = \text{median}\{r_j^{(1)}, r_j^{(2)}, r_j^{(3)}\}, \quad j = 1, 2, \ldots, n
$$

**破坏点理论**：
中位数估计器的破坏点为：

$$
\epsilon^* = \frac{1}{2}
$$

即最多50%的方法失效，仍能给出可靠结果。

**渐近性质**：
当各方法独立时：

$$
\sqrt{K}(\mathbf{r}^{(median)} - \mathbf{r}^{(true)}) \xrightarrow{d} \mathcal{N}\left(\mathbf{0}, \frac{\pi}{2} \boldsymbol{\Sigma}\right)
$$

其中 $K$ 是方法数量，$\boldsymbol{\Sigma}$ 是单个方法的协方差矩阵。

## 5. 多基因座信息融合数学框架

### 5.1 Fisher信息融合理论

**单基因座Fisher信息矩阵**：

$$
\mathcal{I}_{\ell}(\mathbf{r}) = \mathbf{G}^{(\ell)T} (\boldsymbol{\Sigma}_{\ell})^{-1} \mathbf{G}^{(\ell)}
$$

**总Fisher信息**：
当基因座独立时：

$$
\mathcal{I}_{\text{total}}(\mathbf{r}) = \sum_{\ell=1}^{L} \mathcal{I}_{\ell}(\mathbf{r})
$$

**Cramér-Rao下界**：

$$
\text{Cov}(\hat{\mathbf{r}}) \succeq \mathcal{I}_{\text{total}}^{-1}(\mathbf{r})
$$

### 5.2 加权融合的数学优化

**基于拟合质量的权重**：

$$
w_{\ell} = \frac{1}{\|\mathbf{G}^{(\ell)} \mathbf{r}^{(\ell)} - \mathbf{h}^{(\ell)}\|_2^2 + \epsilon}
$$

**加权估计**：

$$
\mathbf{r}^{(weighted)} = \frac{\sum_{\ell=1}^{L} w_{\ell} \mathbf{r}^{(\ell)}}{\sum_{\ell=1}^{L} w_{\ell}}
$$

**最优权重理论**：
最小方差无偏估计的权重为：

$$
w_{\ell}^{(opt)} \propto \text{tr}(\mathcal{I}_{\ell}(\mathbf{r}))
$$

## 6. 误差分析与理论界

### 6.1 观测误差传播分析

**误差模型**：

$$
\mathbf{h}^{(\ell)} = \mathbf{G}^{(\ell)} \mathbf{r}_{\text{true}} + \boldsymbol{\epsilon}^{(\ell)}
$$

其中 $\boldsymbol{\epsilon}^{(\ell)} \sim \mathcal{N}(\mathbf{0}, \sigma_{\ell}^2 \mathbf{I})$

**一阶误差传播**：

$$
\mathbb{E}[\|\mathbf{r}^{(\ell)} - \mathbf{r}_{\text{true}}\|_2^2] \leq \sigma_{\ell}^2 \cdot \text{tr}((\mathbf{G}^{(\ell)T} \mathbf{G}^{(\ell)})^{-1})
$$

**条件数界**：

$$
\|\mathbf{r}^{(\ell)} - \mathbf{r}_{\text{true}}\|_2 \leq \kappa(\mathbf{G}^{(\ell)}) \frac{\|\boldsymbol{\epsilon}^{(\ell)}\|_2}{\|\mathbf{G}^{(\ell)} \mathbf{r}_{\text{true}}\|_2}
$$

### 6.2 融合误差的数学分析

**中位数融合的误差界**：
设单个估计的误差为 $e_{\ell} = \|\mathbf{r}^{(\ell)} - \mathbf{r}_{\text{true}}\|_2$

当 $e_{\ell}$ 独立同分布时：

$$
\mathbb{P}(e_{\text{median}} \leq t) \geq 1 - 2\exp\left(-\frac{Lt^2}{2\sigma^2}\right)
$$

**最优融合的理论下界**：

$$
\text{MSE}_{\min} = \mathbf{r}_{\text{true}}^T \left(\sum_{\ell=1}^{L} \mathcal{I}_{\ell}(\mathbf{r}_{\text{true}})\right)^{-1} \mathbf{r}_{\text{true}}
$$

## 7. 算法复杂度的数学分析

### 7.1 计算复杂度

**NNLS算法**：

- 时间复杂度：$O(m^2 n + mn^2)$
- 空间复杂度：$O(mn)$

**SVD伪逆**：

- 时间复杂度：$O(mn^2 + n^3)$（当 $m \geq n$）
- 空间复杂度：$O(mn + n^2)$

**岭回归**：

- 时间复杂度：$O(n^3)$（矩阵求逆主导）
- 空间复杂度：$O(n^2)$

### 7.2 总复杂度分析

对于 $L$ 个基因座，$n$ 个贡献者：
**总时间复杂度**：

$$
T_{\text{total}} = O(L \cdot \max(m^2 n, n^3))
$$

**总空间复杂度**：

$$
S_{\text{total}} = O(L \cdot mn + n^2)
$$

## 8. 数值稳定性的数学保证

### 8.1 条件数控制策略

**稳定性判据**：

$$
\begin{cases}
\kappa(\mathbf{G}) < 10^{12} & \text{使用NNLS} \\
10^{12} \leq \kappa(\mathbf{G}) < 10^{16} & \text{使用岭回归} \\
\kappa(\mathbf{G}) \geq 10^{16} & \text{使用伪逆}
\end{cases}
$$

### 8.2 正则化参数的理论选择

**L-curve方法**：

$$
\alpha_{\text{opt}} = \arg\min_{\alpha} \mathcal{C}(\alpha) = \|\mathbf{G} \mathbf{r}_{\alpha} - \mathbf{h}\|_2^2 + \|\mathbf{r}_{\alpha}\|_2^2
$$

**广义交叉验证 (GCV)**：

$$
\alpha_{\text{opt}} = \arg\min_{\alpha} \frac{\|\mathbf{A}_{\alpha} \mathbf{h} - \mathbf{h}\|_2^2}{[\text{tr}(\mathbf{I} - \mathbf{A}_{\alpha})]^2}
$$

其中 $\mathbf{A}_{\alpha} = \mathbf{G} (\mathbf{G}^T \mathbf{G} + \alpha \mathbf{I})^{-1} \mathbf{G}^T$

## 9. 性能评估的数学指标

### 9.1 准确性指标

**平均绝对误差**：

$$
\text{MAE} = \frac{1}{n} \sum_{j=1}^{n} |r_j^{(\text{est})} - r_j^{(\text{true})}|
$$

**均方根误差**：

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (r_j^{(\text{est})} - r_j^{(\text{true})})^2}
$$

**决定系数**：

$$
R^2 = 1 - \frac{\sum_{j=1}^{n} (r_j^{(\text{est})} - r_j^{(\text{true})})^2}{\sum_{j=1}^{n} (r_j^{(\text{true})} - \bar{r}^{(\text{true})})^2}
$$

### 9.2 相关性指标

**Pearson相关系数**：

$$
\rho_P = \frac{\sum_{j=1}^{n} (r_j^{(\text{est})} - \bar{r}^{(\text{est})})(r_j^{(\text{true})} - \bar{r}^{(\text{true})})}{\sqrt{\sum_{j=1}^{n} (r_j^{(\text{est})} - \bar{r}^{(\text{est})})^2 \sum_{j=1}^{n} (r_j^{(\text{true})} - \bar{r}^{(\text{true})})^2}}
$$

**Kendall's τ (排序准确性)**：

$$
\tau = \frac{2}{n(n-1)} \sum_{1 \leq i < j \leq n} \text{sgn}(r_i^{(\text{true})} - r_j^{(\text{true})}) \cdot \text{sgn}(r_i^{(\text{est})} - r_j^{(\text{est})})
$$

## 10. 实验结果的数学验证

### 10.1 统计显著性检验

**Kolmogorov-Smirnov检验**：
检验误差分布的正态性：

$$
D_n = \sup_x |F_n(x) - F_0(x)|
$$

**Anderson-Darling检验**：

$$
A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} (2i-1)[\ln F_0(X_i) + \ln(1-F_0(X_{n+1-i}))]
$$

### 10.2 置信区间计算

**Bootstrap置信区间**：
对于估计 $\hat{r}_j$，$(1-\alpha)$ 置信区间为：

$$
\left[\hat{r}_{j,\alpha/2}^{(B)}, \hat{r}_{j,1-\alpha/2}^{(B)}\right]
$$

其中 $\hat{r}_{j,q}^{(B)}$ 是Bootstrap分布的 $q$ 分位数。

**渐近置信区间**：
基于Fisher信息：

$$
\hat{r}_j \pm z_{\alpha/2} \sqrt{[\mathcal{I}^{-1}(\hat{\mathbf{r}})]_{jj}}
$$

## 11. 数学模型的创新总结

### 11.1 理论突破

1. **线性化创新**：

   $$
   \text{非线性生物问题} \xrightarrow{\text{基因型编码}} \text{线性代数问题}
   $$
2. **约束处理**：

   $$
   \mathbf{G} \mathbf{r} = \mathbf{h} + \{\mathbf{r} \geq \mathbf{0}, \mathbf{1}^T \mathbf{r} = 1\}
   $$
3. **稳健融合**：

   $$
   \text{median}\{\text{NNLS}, \text{Pinv}, \text{Ridge}\} \xrightarrow{\text{多基因座}} \text{median}
   $$

### 11.2 数学优势

**全局最优性**：

$$
\mathbf{r}^* = \arg\min_{\mathbf{r} \geq \mathbf{0}} \|\mathbf{G} \mathbf{r} - \mathbf{h}\|_2^2
$$

**数值稳定性**：

$$
\epsilon^* = 50\% \text{ (破坏点)}
$$

**计算效率**：

$$
O(n^3) \ll O(\text{迭代方法})
$$
