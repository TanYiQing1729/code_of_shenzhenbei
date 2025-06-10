
# 机器学习模型选择与性能分析

## 初始模型：逻辑回归基准

**选择理由：**
作为STR多人身份鉴定问题的初始探索，我们选择逻辑回归作为基准模型。逻辑回归具有良好的数学解释性，能够为后续复杂模型的性能提供对比基准，同时其线性决策边界特性有助于理解特征与贡献者人数之间的基本关系。

**STR身份鉴定的数学建模：**
设STR样本经特征工程后的特征向量为$\mathbf{x} = [x_1, x_2, ..., x_{480}]^T$，其中包含16个STR标记提取的特征。对于贡献者人数分类$y \in \{1,2,3,4,5\}$，多项式逻辑回归模型为：

$$
P(y=k|\mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^{4} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}
$$

其中特征向量$\mathbf{x}$具体包含：

**等位基因计数特征：**

$$
\mathbf{x}_{allele} = [count_{D3S1358}, count_{vWA}, ..., ol\_count_{FGA}]
$$

**峰高统计特征：**

$$
\mathbf{x}_{height} = [mean_{D3S1358}, cv_{D3S1358}, ..., kurt_{FGA}]
$$

**全局整合特征：**

$$
\mathbf{x}_{global} = [total\_alleles, ol\_ratio, diversity]
$$

**决策函数的线性假设：**
逻辑回归假设贡献者人数与特征间存在线性关系：

$$
\log\frac{P(y=k|\mathbf{x})}{P(y=1|\mathbf{x})} = \mathbf{w}_k^T \mathbf{x} + b_k
$$

**发现的局限性：**
逻辑回归在STR数据上表现出明显的局限性，准确率仅达到84.62%。主要问题在于STR数据的非线性特征关系：

1. **等位基因组合的非线性模式**：理论上2人混合样本在某标记上可能出现2-4个等位基因，但实际的$count_j$与$y$关系为：

   $$
   \mathbb{E}[count_j|y=n] \neq 2n
   $$

   这种非线性关系无法用$\mathbf{w}^T \mathbf{x}$准确建模。
2. **峰高比值的复杂分布**：混合样本中的峰高比值$ratio_{12} = \frac{h_1}{h_2}$呈现多模态分布，线性组合无法捕获其与贡献者人数的复杂关系。
3. **特征交互的缺失**：STR标记间存在连锁不平衡，需要建模$count_i \times cv_j$类型的交互项，但逻辑回归无法自动发现这些交互关系。

## 第一次优化：梯度提升决策树（GBDT）

**模型转换理由：**
针对STR数据中等位基因计数与贡献者人数的非线性关系，我们引入GBDT。该模型能够自动学习复杂的决策规则，如"当$count_{D3S1358} \geq 3$且$cv_{D3S1358} \geq 0.5$且$ol\_ratio_{D3S1358} > 0.1$时，倾向于预测多人混合"。

**STR数据的GBDT建模：**
GBDT通过加法模型组合多个决策树来处理STR特征的复杂模式：

$$
F_M(\mathbf{x}) = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})
$$

每个弱学习器$h_m(\mathbf{x})$是基于STR特征的决策树，能够学习如下类型的规则：

**第1棵树（基础等位基因模式）：**

```
if count_D3S1358 ≥ 3:
    if cv_D3S1358 ≥ 0.4:
        prediction += 0.8  # 倾向于多人混合
    else:
        prediction += 0.3
else:
    prediction += -0.5  # 倾向于单人样本
```

**第2棵树（OL等位基因模式）：**

```
if ol_ratio_total ≥ 0.15:
    if diversity ≥ 0.6:
        prediction += 0.6  # 强化多人混合判断
```

**残差学习过程：**
对于多分类STR问题，第$m$轮的残差计算为：

$$
r_{im,k} = -\left[\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1,k}(\mathbf{x}_i)}\right]
$$

其中$L$为多分类交叉熵损失：

$$
L = -\sum_{i=1}^{n} \sum_{k=1}^{4} \mathbb{I}[y_i=k] \log P(y=k|\mathbf{x}_i)
$$

**STR特征的自动交互发现：**
GBDT能够自动发现重要的特征组合，如：

- $count_{vWA} \times peak1_{D21S11}$的交互效应
- $ol\_total \times diversity$的协同作用

**性能表现：**
GBDT在STR身份鉴定任务上取得了92.31%的优异准确率，成功捕获了"高等位基因数+高峰高变异+OL等位基因存在→多人混合"的复杂判断逻辑，相比逻辑回归提升。

## 第二次尝试：XGBoost优化

**升级动机：**
考虑到STR数据的高维稀疏特性（480维特征空间），我们尝试XGBoost的正则化优势来防止过拟合，同时利用其二阶梯度信息提高对STR复杂模式的拟合精度。

**STR数据的XGBoost建模：**
XGBoost通过二阶泰勒展开优化目标函数：

$$
Obj^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)
$$

其中正则化项：

$$
\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

**针对STR数据的正则化意义：**

- **$\gamma T$项**：控制决策树叶子节点数量，防止对单个STR标记的微小变化过度拟合
- **$\lambda \sum w_j^2$项**：约束叶子节点权重，避免对异常峰高值过度敏感

**二阶梯度在STR数据上的应用：**

$$
g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}
$$

目标函数近似为：

$$
Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t)
$$

**异常表现分析：**
XGBoost仅达到84.62%准确率，与逻辑回归持平，远低于GBDT。深入分析发现：

1. **过度正则化问题**：STR法医数据样本量相对有限，XGBoost的默认正则化参数过于保守，限制了对关键STR模式的学习。例如，重要的决策规则"$count_{D3S1358} = 4$ AND $cv_{D3S1358} > 0.6$ → $y=2$"可能因正则化被抑制。
2. **特征选择过于严格**：XGBoost基于增益的分裂标准可能错过STR数据中重要但信号微弱的模式，如某些标记的$ol\_ratio_j$虽然数值较小但对混合检测很重要。
3. **小样本二阶优化不稳定**：在STR数据的小样本环境下，二阶导数$h_i$的估计不够稳定，导致优化过程震荡，影响收敛到最优解。

## 概率模型验证：高斯朴素贝叶斯

**STR数据的概率建模：**
将STR特征向量分解为$\mathbf{x} = [\mathbf{x}_{allele}, \mathbf{x}_{height}, \mathbf{x}_{global}]$，朴素贝叶斯模型假设特征条件独立：

$$
P(y=c|\mathbf{x}) = \frac{P(y=c) \prod_{i=1}^{480} P(x_i|y=c)}{P(\mathbf{x})}
$$

**对连续STR特征的高斯假设：**
对于峰高统计特征$mean_j$，假设在给定贡献者人数$c$下服从高斯分布：

$$
P(mean_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left(-\frac{(mean_j-\mu_{c,j})^2}{2\sigma_{c,j}^2}\right)
$$

**STR数据的独立性假设严重违反：**
实际STR数据中存在强相关性：

1. **同标记内特征相关**：$count_{D3S1358}$与$cv_{D3S1358}$存在强正相关
2. **跨标记特征依赖**：由于连锁不平衡，不同标记的等位基因计数并非独立
3. **全局与局部特征关联**：$total\_alleles = \sum_{j=1}^{16} count_j$，显然违反独立性假设

**预期性能限制：**
由于严重的独立性假设违反，朴素贝叶斯达到61.54%的准确率符合预期。

## 深度学习探索：多层感知机（MLP）

**STR特征的层次化表示学习：**
设计递减式网络结构$480 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 4$，模拟STR身份鉴定专家的推理层次：

**第一隐藏层（64维）- 基础STR特征组合：**

$$
\mathbf{h}^{(1)} = \sigma(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)})
$$

学习基础的标记内特征组合，如$f_1(count_j, cv_j)$用于识别单标记的混合模式。

**第二隐藏层（32维）- 标记间模式识别：**

$$
\mathbf{h}^{(2)} = \sigma(W^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})
$$

整合多个标记信息，学习跨标记模式。

**第三隐藏层（16维）- 高级判断特征：**

$$
\mathbf{h}^{(3)} = \sigma(W^{(3)}\mathbf{h}^{(2)} + \mathbf{b}^{(3)})
$$

提取抽象的判断特征，类似专家综合所有信息后的高级推理。

**输出层 - 贡献者人数概率：**

$$
P(y=k|\mathbf{x}) = softmax(W^{(4)}\mathbf{h}^{(3)} + \mathbf{b}^{(4)})_k
$$

**STR数据的批归一化必要性：**
不同STR标记的峰高范围差异极大，批归一化处理这种分布差异：

$$
\hat{x}_{mean_j} = \frac{x_{mean_j} - \mu_{batch}}{\sqrt{\sigma_{batch}^2 + \epsilon}}
$$

**针对STR数据的Dropout策略：**
使用较低的dropout率0.1保留关键信息：

$$
h_i^{(l)} = \begin{cases} 
0 & \text{概率为0.1} \\
\frac{h_i^{(l)}}{0.9} & \text{概率为0.9}
\end{cases}
$$

**性能表现与解释：**
MLP的准确率存在波动，但是整体上在80%到95%之间，性能尚可。
