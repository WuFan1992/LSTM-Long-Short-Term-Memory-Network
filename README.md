# LSTM-Long-Short-Term-Memory-Network

## RNN 存在的问题


RNN 误差传递的公式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/1.PNG)

我们关注模的边界，因为模可以反映误差的大小

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/2.PNG)

因为存在指数，所以Bf 和 Bw 的乘积会迅速指数增长，或者迅速衰减，我们来看指数衰减的情况

根据前面的知识，对于一条s链，也就是一个隐藏层来说，它的权重数组是各个时刻的权重数组的和，假设一条s链有6个时刻，那么权重数组就是这6个时刻的和
公式如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/3.PNG)

画成图我们来看

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/4.png)
