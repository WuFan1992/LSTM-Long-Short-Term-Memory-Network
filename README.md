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
