# LSTM-Long-Short-Term-Memory-Network

## RNN 存在的问题


RNN 误差传递的公式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/1.PNG)

我们关注模的边界，因为模可以反映误差的大小

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/2.PNG)

因为存在指数，所以Bf 和 Bw ，分别代表对角net 和 权重矩阵W 的模的最大值，它们的乘积会迅速指数增长，或者迅速衰减，我们来看指数衰减的情况

根据前面的知识，对于一条s链，也就是一个隐藏层来说，它的权重数组是各个时刻的权重数组的和，假设一条s链有6个时刻，那么权重数组就是这6个时刻的和
公式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/3.PNG)

画成图我们来看

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/4.png)


由于某一时刻的权重矩阵，和当前时刻的误差有关，所以权重矩阵也随着时刻的变化而不断变化，这里是衰减的情况。
我们注意到，t-3时刻之后，权重就几乎约等于0 了，也就是说，t-3时刻之后，不管t-3 时刻之后的状态量h 如何，它都不会对最终的权重矩阵有影响，换句话说，只有t，t-1，t-2，t-3 对最终的权重有所影响，所以可以说，RNN 对短时的输入敏感，但是对于长时的输入，却不敏感，为了能够让长时，也就是说为了能够时t-4 t-5 之后的时刻的状态量h，也能够对最终权重矩阵有影响我们，需要保留前面的梯度。于是乎就有了LSTM

## LSTM 的基本构成

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/5.png)

在 这幅图中，C用来保存长期状态，xt 是当前的输入， h是邻近的状态（ht-1）是上一时刻的状态

为了能控制C xt h ,我们设立了三个门

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/6.png)

门的本质就是**全连接层**，也就是说对于一个加权输入，有一个激励输出，门的公式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/7.PNG)

我们把C 称之为单元状态state ,所以三个门分别控制：
1. 遗忘门 forget gate 控制前一个state 对当前state 的影响
2. 输入门 input gate 控制当前的输入xt 对当前state 的影响
3. 输出门 output gate 控制当前state对当前输出ht 的影响

**遗忘门**

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/8.PNG)


[ht-1,xt]是把输入向量和状态向量连接在一起，这样组合起来的维度就是二者的维度之和。假如输入向量xt 的维度是3000 ，状态向量ht的维度是1500，那么[ht-1,xt]的维度就是4500。根据上一章 RNN的知识，状态向量ht 在同一个隐藏层的维度不变，同样，输入向量xt 在同一个隐藏层，也就是同一个s链上维度不变。
再来看Wf 的维度。我们回顾一下RNN，回忆如下公式:

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/20.PNG)

在上面的公式里，x 是输入向量，s 是前一时刻的状态量，和遗忘门里面的h 是一致的，s 权重矩阵的维度是NXN。所以类比过来，以上面的具体数值为例子，那么Wf 的维度就是4500 x 4500
从图像中来看，ht 是横向传播，而xt 是纵向传播。回忆RNN 的知识，我们知道误差在反向传播的时候，也存在着纵向传播（隐藏层之间的传播），和横向传播（时刻之间的传播），这两者相互独立，为了能够在后面求偏导数的时候方便，我们把Wf 拆成 Wh 和 Wx。拆开后的公式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/9.PNG)

遗忘门的图例如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/10.png)


**输入门**

一样的道理，输入门的表达式如下

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/11.PNG)

图例如下

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/12.png)


ct（波浪号)是当前输入的单元状态state ,而ct 是当前的时刻的单元状态state，前者针对输入，后者针对时刻。

ct(波浪号)的表达式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/13.PNG)
