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
## LSTM 正向传递

我们观察到，大部分的正向传递都是这样的形式

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/8.PNG)


所以我们首先定义一个calcul_gate 函数，有了这个函数，可以简便计算

```
def calcul_gate(self,Wx,Wh,b,input,function):

    h_before = self.h_list(self.times -1)

    gate = (np.dot(Wx,h_before) + np.dot(Wx, input))+ b

    output = self.sigmoid(gate)

    return output
```

接着就是主体的正向传播函数

```
def forward_LSTM(self,x):

    self.times +=1

    # forget gate output
    f_gate = calcul_gate(self.Wfx,self.Wfh,self.bf,x,self.sigmoid)
    self.f_list.append(f_gate)

    # input gate output
    i_gate = calcul_gate(self.Wix,self.Wih,self.bi,x,self.sigmoid)
    self.i_list.append(i_gate)

    # cell state of input
    ct_gate = calcul_gate(self.Wcx,self.Wch,self.bc,x,self.tahn)
    self.ct_list.append(ct_gate)

    # cell state
    c_gate = f_gate[self.times]*ct_list[self.times-1] + i_gate[self.times]*ct_gate[self.times]

    # output gate
    o_gate = calcul_gate(self.Wox,self.Woh,self.bo,x,self.sigmoid)
    self.o_list.append(o_gate)

    # condition gate
    h_gate = o_list[self.times] * self.tanh(c_gate[self.times])
    h_list.append(h_gate)
   
```

## LSTM 反向传递
### 误差的反向传递
#### 误差沿纵向（时间）反向传递
反向传递的代码思路如下：
1. 首先找出第l-1 层的误差，和第 l 层的误差之间的数学关系
2. 然后先定义一个从K+1 层 到K 层误差传递的一个函数
3. 再根据节点数，用一个for 循环不断循环2 中的函数，最后把所有误差都存进一个列表里
4. 3种得到的误差将会在权重更新当中获得应用

根据forget gate  input gate output gate cell state 的关系，我们知道i,o,f,ct 都和ht-1相关，为什么这么说，因为：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/24.PNG)

省略推导过程，直接得到第l-1 层误差和第l 层误差的关系

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/25.PNG)

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/26.PNG)

由此可见，上一层的误差和deltat_f deltat_i deltat_o deltat_c 都有关，所以在代码里

```
def calcul_deltat_k(self,k):

    fg = self.f_list[k]
    ig = self.i_list[k]
    cg = self.ct_list[k]
    og = self.o_list[k]

    deltat_k = self.deltat_list[k]

    deltat_o = deltat_k * self.tanh(cg) * og * (1-og)

    deltat_f = deltat_k * og * (1-self.tanh(cg)*self.tanh(cg)) * self.c_list[k-1]*fg*(1-fg)

    deltat_i = deltat_k * og * (1-self.tanh(cg)*self.tanh(cg)) * cg * ig * (1-ig)

    deltat_c = deltat_k * og * (1-self.tanh(cg)*self.tanh(cg)) *ig * (1-cg*cg)

    detat_together = (np.dot(deltat_o,self.Woh) + np.dot(deltat_f,self.Wfh) + np.dot(deltat_c,self.Wch) + np.dot(deltat_i,self.Wih)).transpose()

    deltat_list[k-1] = deltat_together

    deltat_i_list[k] = deltat_i
    deltat_o_list[k] = deltat_o
    deltat_f_list[k] = deltat_f
    deltat_c_list[k] = deltat_c
```

**关键点** 无论是RNN 还是LSTM 同一个隐藏层，所有输入共用一个权重矩阵，所有状态量h 共用一个权重矩阵
在LSTM 这里，也就是遗忘门的所有时刻（横向）共用一个权重矩阵Wfh，输入门的所有时刻共用一个权重矩阵Wih，输出门的所有时刻共用一个权重矩阵Woh，单元状态的所有时刻共用一个权重矩阵Wch

接着初始化用来储存每一层deltat_list的向量：
```
def init_deltat(self):

    deltat_list = []

    for i in range(self.times +1):
        deltat_list.append(np.zeros((self.state_width,1)))
    return deltat_list

```
最后利用一个for 循环包括一个隐藏层当中的所有节点

```
def calcul_deltat(self,deltat_h):

    deltat_i_list = self.init_deltat()
    deltat_o_list = self.init_deltat()
    deltat_c_list = self.init_deltat()
    deltat_f_list = self.init_deltat()
    deltat_ct_list = self.init_deltat()
    deltat_list[-1] = deltat_h

    for i in range(self.times-1,0,-1):

        calcul_deltat_k(i)
```
**关键点**

上面的代码里，我们在初始化deltat_list 时，长度不是self,times 而是self.times+1 ，这是因为这里是横向沿时间方向传递，但是同时也存在着从上一层传递下来的情况，所以在deltat_list[-1]里存储上一层传递下来（纵向传递）
接下来在计算权重和偏置更新时，由于我们计算的是横向时间方向的更新，所以需要一个层间纵向传递的误差参数


### 权重，偏置的更新

神经网络最终的目的就是权重和偏置更新，基本的思路都是先求出误差的传递，然后把每一层的误差保存进一个list里，在求解权重和偏置的偏导数时，就需要用到每一层的误差。
注意权重矩阵有两个，一个是Wh ，另一个是Wx
我们先来看Wh 的偏导数，某一时刻t的偏导数：

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/27.PNG)

最终的梯度是各个时刻的权重梯度加在一起，公式如下

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/28.PNG)

同样对于偏置，也是一样，某一时刻t 的偏置是:

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/29.PNG)

最终的偏置是各个时刻偏置梯度的加在一起

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/30.PNG)

以上是横向的，因为横向有多个时刻，所以需要相加，但是 对于纵向的传递，由于一次只传递一层，所以不存在加和的情况
公式如下

![](https://github.com/WuFan1992/LSTM-Long-Short-Term-Memory-Network/blob/master/image/31.PNG)

代码部分，首先也是求从t时刻到t-1时刻， 权重梯度

```
def calcul_gradient_k(self,t):

    '''
    we calcul here in order to get the result at time t


    '''
    h_pre = h_list[t-1].transpose()
    gradient_Woh = np.dot(self.deltat_o_list[t],h_pre)
    gradient_Wfh = np.dot(self.deltat_f_list[t],h_pre)
    gradient_Wih = np.dot(self.deltat_i_list[t],h_pre)
    gradient_Wch = np.dot(self.deltat_ct_list[t],h_pre)

    gradient_bf= self.deltat_f_list[t]
    gradient_bi = self.deltat_i_list[t]
    gradient_bc = self.deltat_c_list[t]
    gradient_bo = self.deltat_o_list[t]

    return gradient_Woh, gradient_Wfh,gradient_Wih,gradient_Wch,gradient_bf,gradient_bi,gradient_bc,gradient_bo
 
 ```

然后，初始化Wh Wx b 三个矩阵
```

def init_gradient_matrix(self):

    '''
    in this part , we have 3 matrix to be initialization
    Wx,Wh,b
    so the return value must be these 3 matrice

    '''
    matrix_Wh = np.randoms.uniform(-1e-4,1e-4,(self.state_width,self.state_width))
    matrix_Wx = np.randoms.uniform(-1e-4,1e-4,(self.state_width,self.input_width))
    matrix_b  = np.zeros((self.state_width,1))

    return matrix_Wh, matrix_Wx, matrix_b
    
    ```
    
    
    最后用一个for 循环汇总
    
    ```
    def calcul_gradient(self,x):

    self.gradient_Woh ,self.gradient_Wox, self.gradient_bo = self.init_gradient_matrix()
    self.gradient_Wih ,self.gradient_Wix, self.gradient_bi = self.init_gradient_matrix()
    self.gradient_Wfh ,self.gradient_Wfx, self.gradient_bf = self.init_gradient_matrix()
    self.gradient_Wch ,self.gradient_Wcx, self.gradient_bc = self.init_gradient_matrix()

    for t in range(self.times,0-1):
        gradient_Woh, gradient_Wfh,gradient_Wih,gradient_Wch,gradient_bf,gradient_bi,gradient_bc,gradient_bo = self.calcul_gradient_k(t)
        self.gradient_Woh +=gradient_Woh
        self.gradient_bo +=gradient_bo
        self.gradient_Wih +=gradient_Wih
        self.gradient_bi +=gradient_bi
        self.gradient_Wfh +=gradient_Wfh
        self.gradient_bf +=gradient_bf
        self.gradient_Wch +=gradient_Wch
        self.gradient_bc +=gradient_bc

    xt = x.transpose()
    self.gradient_Wfx = np.dot(self.deltat_f_list[-1],xt)
    self.gradient_Wcx = np.dot(self.deltat_c_list[-1],xt)
    self.gradient_Wix = np.dot(self.deltat_i_list[-1],xt)
    self.gradient_Wox = np.dot(self.deltat_o_list[-1],xt)
    
 ```
 
 **关键点**
 
 因为 Wh 涉及到加和，所以需要一个for 循环。而Wx 由于不需要加和，所以放到了for 循环的外面。
 
 
    
