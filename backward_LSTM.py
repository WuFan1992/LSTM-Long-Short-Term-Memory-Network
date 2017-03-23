import numpy as np
import class_LSTMLayer
import forward_LSTM

# define the backward funtion in the areas of time


def init_deltat(self):

    deltat_list = []

    for i in range(self.times +1):
        deltat_list.append(np.zeros((self.state_width,1)))
    return deltat_list

        

def calcul_deltat_k(self,k):

    fg = self.f_list[k]
    ig = self.i_list[k]
    cg = self.ct_list[k]
    og = self.o_list[k]
    ctg = self.ct_list[k]

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


def calcul_deltat(self,deltat_h):

    deltat_i_list = self.init_deltat()
    deltat_o_list = self.init_deltat()
    deltat_c_list = self.init_deltat()
    deltat_f_list = self.init_deltat()
    deltat_ct_list = self.init_deltat()
    deltat_list[-1] = deltat_h

    for i in range(self.times-1,0,-1):

        calcul_deltat_k(i)



    
