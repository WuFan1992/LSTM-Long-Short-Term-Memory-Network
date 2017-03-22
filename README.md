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

它的图像如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/14.png)

而 ct 的表达式如下

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/22.PNG)

它由输入门，乘以输入单元状态ct（波浪号），加上遗忘门，乘以前一时刻的 单元状态量ct-1

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/15.png)

其中的圆圈代表的是按元素相乘，举例如下：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/21.PNG)



**输出门**

输出门的表达式为：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/16.PNG)

示意图如下所示：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/17.png)


而最终LSTM输出的表达式为：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/18.PNG)


最终输出的图片如下

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/19.png)

以上的公式同时也是前向 传递的公式

下面来看代码部分

思路一：先定义激活函数，因为在LSTM里面涉及到两个激活函数sigmoid 和 tahn 函数，我们先来看这两个函数的定义（用于正向传递）和导数（导数用于反向传递）

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/23.PNG)


```
#define the two activation function

def sigmoid(self,weight_input):

    return 1.0 / (1.0+np.exp(-1*weight_input))


def sigmoid_derivative(self,output):

    x = sigmoid(output)

    return x*(1-x)


def tanh(self,weight_input):

    a = exp(weight_input)

    b = exp(-1*weight_input)

    return (a-b)/(a+b)


def tahn_derivative(self,output):

    c = tahn(output)

    return 1-c^2
 ```
 
 **重点提示** exp 是numpy 包内的函数，所以一定要使用np.exp
 
 
 思路二：定义LSTRLayer类
 
 1. 先考虑输入 输入x 我们定义一个input_width　代表x的维度
 2. 中间的状态量h 我们定义一个state_width 代表h 的维度
 3. 用于存放一个隐藏层里，遗忘门的列表f_list， 输入门的列表i_list 输出门的列表 o_list  -----三个门的列表
    
    状态量的列表 h_list ,当前输入的单元状态列表ct_list ，单元状态列表 c_list
 4. 遗忘门，输入门， 输出门，单元状态的权重矩阵和对应的偏置矩阵
 
```
class LSTMLayer(object):

    def _init_(self, input_width,state_width,learning_rate):

        self.input_width = input_width

        self.state_width = state_width

        self.learning_rate = learning_rate
        
        #forget gate
        self.f_list = self.init_vector()

        #input gate
        self.i_list = self.init_vector()

        #output gate
        self.o_list = self.init_vector()

        #condition
        self.h_list = self.init_vector()

        # cell state of input
        self.ct_list = self.init_vevtor()

        #cell state
        self.c_list = self.init_vector()

        # initialization of forget weight vector
        Wfx, Wfh, bf = self.init_matrix()

        # initialization of input weight vector
        Wix , Wih, bi = self.init_matrix()

        # initialization of output weight vector
        Wox , Woh , bo = self.init_matrix()

        # initialization of cell state weight vector
        Wcx ,Wch, bc = self.init_matrix()

        # at last
        self.times = 0

        




    def init_vector(self):

        init_state_vector = []

        init_state_vector.append(np.zeros((self.state_width,1)))

        return init_state_vector


    def init_matrix(self):

        Wx = np.randoms.uniform(-1e-4,1e-4,(self.state_width,self.input_width))

        Wh =  np.randoms.uniform(-1e-4,1e-4,(self.state_width,self.state_width))

        bf = np.zeros((self.state_width.1))

        return Wx, Wh, bf


```
 
 
 
