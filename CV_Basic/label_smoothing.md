## <center> label smoothing </center>

Labeling smoothing 在深度学习中通常作为一种正则化技术（regularization techniques）避免模型在训练过程中的 overfitting。例如在分类任务中，有时模型在学习训练数据时过于 confident，从而导致模型的泛化能力不足。

### Background

在分类任务中，假设目标有 `K` 个类别 {1, 2, ... K}，对于训练数据 $i$, $(x_i, y_i)$，其中 $x_i$ 为输入，$y_i$ 为对应的标签。因此，我们可以获得一个输入数据在所有类别上的 ground truth distribution $p(y|x_i)$，且$\sum_{y=1}^{K} p(y|x_i) = 1$。在训练过程中，我们希望模型（参数为 $\theta$）的输出 $q_\theta(y|x_i)$ 能够尽可能的接近 ground truth distribution $p(y|x_i)$，即 $q_\theta(y|x_i) \approx p(y|x_i)$。在分类任务中，我们通常使用 cross entropy loss 来衡量 $q_\theta(y|x_i)$ 和 $p(y|x_i)$ 之间的差异，即：$$H(p, q_\theta) = - \sum_{y=1}^{K} p(y|x_i) \log q_\theta(y|x_i)$$

如果训练数据集中有 $n$ 个数据，那么我们可以得到整个训练数据集上的 cross entropy loss：$$\mathcal{L} = \sum_{i=1}^{n} H(p, q_\theta) = - \sum_{i=1}^{n} \sum_{y=1}^{K} p(y|x_i) \log q_\theta(y|x_i)$$

#### One-Hot Labeling  
通常，对于 $p(y|x_i)$，我们使用 one-hot encoding 来表示，即 $$p(y|x_i) = \begin{cases} 1, & \text{if } y = y_i \\ 0, & \text{otherwise} \end{cases}$$


因此，$\mathcal{L}$ 可以简化为：
$$
\begin{aligned}
\mathcal{L} &= \sum_{i=1}^{n} H(p, q_\theta) \\ &= - \sum_{i=1}^{n} \sum_{y=1}^{K} p(y|x_i) \log q_\theta(y|x_i) \\ &= - \sum_{i=1}^{n} p(y_i|x_i) \log q_\theta(y_i|x_i) \\ &= - \sum_{i=1}^{n} \log q_\theta(y_i|x_i)
\end{aligned}
$$

其中，$q_\theta(y_i|x_i)$ 通常通过 softmax 函数来计算，即：$$q_\theta(y_i|x_i) = \frac{\exp(z_{y_i})}{\sum_{j=1}^{K} \exp(z_j)}$$，其中 $z_j$ 为模型在第 $j$ 个类别上的 logit。

在训练过程中，我们通过最小化 $\mathcal{L}$ 使得 $q_\theta(y_i|x_i)$ 尽可能的接近 $p(y_i|x_i)$。（该过程等价于在训练数据集上进行最大似然估计 （证明见 [MLE](#最大似然估计-maximun-likelihood-estimation)）。随着训练的进行，$\mathcal{L}$ 会逐渐减小，且 $\rightarrow 0$, 当 $q_\theta(y_i|x_i) \rightarrow 1$ 时，$q_\theta(y_j|x_i) \rightarrow 0, \forall j \neq i$，也即是说，模型在训练过程中会逐渐变得 confident，从而导致模型的泛化能力不足。

### Label Smoothing
为了避免模型在训练过程中的 over confident，我们可以对 ground truth distribution $p(y|x_i)$ 进行平滑处理。具体来说，我们引入一个噪声分布（noise distribution） $u(y|x_i)$，并将 $p(y|x_i)$ 和 $u(y|x_i)$ 进行加权平均，即：
$$
\begin{aligned}
p'(y|x_i) &= (1 - \epsilon) p(y|x_i) + \epsilon u(y|x_i)  \\
&= \begin{cases} 1 - \epsilon +  \epsilon u(y|x_i), & \text{if } y = y_i \\ \epsilon u(y|x_i), & \text{otherwise} \end{cases}
\end{aligned}
$$
其中 $\epsilon$ 为平滑参数，$\epsilon \in [0, 1]$。当 $\epsilon = 0$ 时，$p'(y|x_i) = p(y|x_i)$，当 $\epsilon = 1$ 时，$p'(y|x_i) = u(y|x_i)$。且 $\sum_{y=1}^{K} p'(y|x_i) = 1$。

我们使用新的 ground truth label $p'(y|x_i)$ 来替代原来的 $p(y|x_i)$，从而得到新的 cross entropy loss：
$$
\begin{aligned}
\mathcal{L}' &= - \sum_{i=1}^{n} \sum_{y=1}^{K} p'(y|x_i) \log q_\theta(y|x_i) \\
&= - \sum_{i=1}^{n} \sum_{y=1}^{K} \left[ (1 - \epsilon) p(y|x_i) + \epsilon u(y|x_i) \right] \log q_\theta(y|x_i) \\
&= - \sum_{i=1}^{n} \left[ (1 - \epsilon) \sum_{y=1}^{K} p(y|x_i) \log q_\theta(y|x_i) + \epsilon \sum_{y=1}^{K} u(y|x_i) \log q_\theta(y|x_i) \right] \\
&= \sum_{i=1}^{n} \left[ (1 - \epsilon) H(p, q_\theta) + \epsilon H(u, q_\theta) \right]  \\
&= \sum_{i=1}^{n} \left[ (1 - \epsilon) \left[-log q_\theta(y_i|x_i)\right] + \epsilon \sum_{y=1}^{K} u(y|x_i) \left[-\log q_\theta(y|x_i) \right]\right] \\
\end{aligned}
$$

可以看出，新的损失函数 $\mathcal{L}'$ 由两部分组成，第一部分为原来的 cross entropy loss，第二部分为噪声分布 $u(y|x_i)$ 和模型预测分布 $q_\theta(y|x_i)$ 之间的 cross entropy loss。当 $\epsilon = 0$ 时，$\mathcal{L}' = \mathcal{L}$，当 $\epsilon = 1$ 时，$\mathcal{L}' = \sum_{i=1}^{n} H(u, q_\theta)$。因此，当 $\epsilon \neq 0$ 时，如果模型的预测过于 confident，即 $q_\theta(y_i|x_i) \rightarrow 1$，$H(p, q_\theta) \rightarrow 0$，而 $H(u, q_\theta)$ 会显著增大，从而导致 $\mathcal{L}'$ 增大，从而减小了模型的 confident。

### 结论
因此，label smoothing 的作用是：在训练过程中，引入了一个额外的正则化项 $H(u, q_\theta)$，从而避免了模型的 over confident，提高了模型的泛化能力。一般来说，我们可以使用均匀分布作为噪声分布 $u(y|x_i)$，即 $u(y|x_i) = \frac{1}{K}$。

### 代码实现 python & torch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(torch.nn.Module):
    """
    Label Smoothing Loss Implementation.

    Args:
        epsilon (float): Smoothing factor.
        reduction (str): Type of reduction to apply to loss ('mean', 'sum', or 'none').
        weight (torch.Tensor, optional): Weight tensor for the loss. Defaults to None.
    
    Note:
        This implementation assumes that the input `predict_tensor` is log-probability. (i.e. `F.log_softmax` or `nn.LogSoftmax` is applied to the input).
    
    References:
        [1] Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
    """
    def __init__(self, epsilon: float, reduction: str = 'mean', weight: torch.Tensor = None):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss: torch.Tensor):
        """
        Reduce the loss tensor based on the specified reduction type.

        Args:
            loss (torch.Tensor): Loss tensor to be reduced.
        
        Returns:
            torch.Tensor: Reduced loss tensor.
        """
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def linear_combination(self, x: torch.Tensor, y: torch.Tensor):
        """
        Linearly combine two tensors `x` and `y` based on the smoothing factor `self.epsilon`.

        Args:
            x (torch.Tensor): First tensor.
            y (torch.Tensor): Second tensor.
        
        Returns:
            torch.Tensor: Linearly combined tensor.
        """
        return (1 - self.epsilon) * x + self.epsilon * y

    def forward(self, predict_tensor, target):
        """
        Forward pass of the label smoothing loss.

        Args:
            predict_tensor (torch.Tensor): Predicted tensor. shape of (batch_size, num_classes)
            target (torch.Tensor): Target tensor. shape of (batch_size, num_classes)

        Returns:
            torch.Tensor: Loss tensor.
        """
        assert 0 <= self.epsilon < 1

        if self.weight is not None:
            self.weight = self.weight.to(predict_tensor.device)
        
        num_classes = predict_tensor.size(-1)

        log_preds = F.log_softmax(predict_tensor, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        negative_log_likelihood = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)

        return self.linear_combination(negative_log_likelihood, loss / num_classes)

```
---
## annex
### 最大似然估计 (Maximun Likelihood Estimation)
最大似然估计是一种参数估计方法，它的基本思想是：选择使得观测数据出现概率最大的参数值作为估计值。在监督学习中，给定数据集 ${(x_1, y_1), (x_2, y_2), ... (x_n, y_n)}$，其中 $x_i$ 为输入，$y_i$ 为对应的标签。我们期望寻找一组参数 $\theta$ 使得数据集上所有样本的可能性的乘积最大，即：$$\theta_{MLE} = \arg \max_{\theta} \prod_{i=1}^{n} q_\theta(y_i|x_i)$$

上式等价于 (等式两边同时取对数)：$$\arg\max_{\theta} \sum_{i=1}^{n} \log q_\theta(y_i|x_i)$$

也即是（最小化 $\mathcal{L}$）：$$\arg\min_{\theta} - \sum_{i=1}^{n} \log q_\theta(y_i|x_i)$$