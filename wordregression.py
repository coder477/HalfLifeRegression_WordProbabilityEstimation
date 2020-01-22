from __future__ import division
import numpy as np
import decimal

import boto3
import time
import decimal
session1 = boto3.Session(profile_name='default')
dynamodb = boto3.resource('dynamodb',region_name='us-east-1',)

class StuWordRegression():
    
    """
    Implements regression to estimate the decay factor of retaining words in memory.
    
    """
    
    def __init__(self, n_features, w=None, lr=0.002, w_h = 0.02, w_l2 = 0.2, sigma = 1, fcounts=None, min_h=0.0104167, max_h=274): 
             
        if w is None:
            # Initialise weights and account for bias term
            self.w = 3*np.random.rand(n_features+1)
        else:
            self.w = w
            assert (len(self.w) == n_features+1), 'number of weights should be equal to number of features + 1'
        self.lr = lr
        self.w_h = w_h
        self.w_l2 = w_l2
        self.sigma = sigma
        if fcounts is None:
            self.fcounts = np.zeros(n_features)
        else:
            self.fcounts = fcounts
            assert (len(self.fcounts) == n_features), 'number of feature counts \
              should be equal to number of features'
        
        self.min_h = min_h
        self.max_h = max_h
        
    def halflife(self, x):        
        theta = np.dot(x, self.w)
        print "theta",theta
        h = 2 ** theta
        h = min(max(h, self.min_h), self.max_h)
        
        return h
    
    def predict(self, x, t):        
        h = self.halflife(x)
        p = 2 ** (-t/h) #delta is t
        # Clip p for numerical conditioning
        p = min(max(p, 0.1), .9)
        
        return p, h
    
    def train(self, x, p, h, t):
        p_hat, h_hat = self.predict(x, t)

        dL_p = 2 * (p_hat-p) * (np.log(2)**2) * p_hat * (t/h_hat)
        #dL_p=format(float(dL_p),'.25f')
        dL_h = 2 * (h_hat-h) * h_hat * np.log(2)
        # dL_h=format(float(dL_h),'.25f')
        for i in range(len(x)-1):
            rate = (1/(1+p)) * self.lr / np.sqrt(1 + self.fcounts[i])
            self.w[i] -= rate * dL_p * x[i]
            
            self.w[i] -= rate * self.w_h * dL_h * x[i]
            self.w[i] -= rate * self.w_l2 * self.w[i] / self.sigma**2
            self.fcounts[i] += 25
        a = ['{:f}'.format(item) for item in self.w]
       
        return a,self.fcounts 

def insertto_ww(z,word_id):
    table = dynamodb.Table('word_weight_vectors')
    
    table.put_item(Item={'word_id':word_id,'w1': decimal.Decimal(z[0]),'w2':decimal.Decimal(z[1]),'w3':decimal.Decimal(z[2]),'w4':decimal.Decimal(z[3]),'w5':decimal.Decimal(z[4])})    

if __name__ == "__main__":
    
    a=StuWordRegression(4, None,0.00009,  0.0001,  0.09,  0.8, None, 0.0104167, 274)
    #x is each row in the data array thru numpy
    #a.train(x, p, h, t)
    
    from pandas import DataFrame, read_csv
    import pandas as pd 
     
    file = r'wpe_data.xls'
    df = pd.read_excel(file)

    mt=df.as_matrix()
    
    
    for each in range(0,len(mt)-1):
        #  print "w", (a.w)
        #for each in range(0,25):
        l=mt[each][1]
        r=mt[each][2]
        c=mt[each][3]
        wr=mt[each][4]
        t=mt[each][5]
          
        x=np.array([l,r,c,wr,0])
        
        print "x", (x)
           
        p=decimal.Decimal((2*c+l)/(2*(c+wr)+l+r))
        p= round(p,5)
    
        if(p==1.0):
            p=0.99
        elif(p==0.0):
            p=0.01
        
        h=-t/(np.log2(p))
    #     h= np.dot(x,a.w)
    #     print "h", h
        z,f= a.train(x,p,h,t)
        print z,mt[each][0]
        insertto_ww(z,mt[each][0])
        
        
    # w= np.array([  6.70134594e+05 ,  3.84983813e-04 ,  3.71181572e-05 ,  9.83216126e-04 ,3.93123739e-04])
    # x = np.array([ 434.6 , 192.4 ,   0. ,    0. ,    0. ])
    # print np.dot(w,x)


