import numpy as np
import backward_LSTM

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
    
    

    
    

    

    

    
