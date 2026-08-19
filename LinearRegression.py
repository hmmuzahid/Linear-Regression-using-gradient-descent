import numpy as np
try:
  from mat_opr_python import *
  import mat_opr_python as obj
except:
  import numpy as obj

class LinearRegression:
  def __init__(self, lambda_ = 0):
    self.lambda_ = lambda_
  
  #Create a method to fit the linear regression model
  def fit(self, x, y, obj=obj, max_iter=500):
    m, n = x.shape
    y = y.reshape(-1, 1)
    
    #Calculate the mean of each features
    means = x.mean(axis=0, keepdims=True)
    
    #Calculate the std of each features
    stds = x.std(axis=0, keepdims=True) + 1e-4
    
    #Subtract the mean and divide by the std
    #It removes the bias(intercept) out of the equation and can reduce the variance
    x = (x - means)/stds
    
    #Normalize target as well
    ymean = y.mean()
    
    ystd = y.std() + 1e-4
    
    y = (y-ymean) / ystd
    
    #Create initial weight(s)
    #Start with 0.1
    w = obj.array([0.1]*n).reshape(-1, 1)
    
    #Create initial rate(s)
    #Start with 0.1
    rate = obj.array([.1]*n).reshape(-1, 1)
    
    #Create initial acceleration for rate(s)
    #We increase the rate(s) by 10% until we find the maximum rate(s) that doesn't overshoot(s)
    accel = obj.array([.1]*n).reshape(-1, 1)
    
    #Keep track of the previous derivative(s) to compare it with the new one(s)
    old_grd = None
    for _ in range(max_iter):
      #Calculate new gradient
      tmp = x @ w - y
      new_grd = obj.array((x.T @ tmp)/m) + (self.lambda_/m)*w
      #Update rate
      rate += rate*accel
      
      #Calculate the very first derivatives and set it to old_grd and continue
      if old_grd is None:
        old_grd = new_grd
        continue
      
      #Use masking to check overshooting
      mask = (new_grd>0) != (old_grd>0)
      #print(mask.shape)
      accel[mask] = 0
      rate[mask] /= 1.1
      w += obj.where(mask , rate*old_grd, -rate*new_grd)
      rate[mask] /= 1.1
      
      old_grd = new_grd
      
      #If any all derivative is less than or equal to 1e-6, we reached the minimum. Return the weights
      if (abs(new_grd) <= 1e-4).all():
        #Scale the weight(s) back to the original form and return them
        
        self.w = obj.array((w * ystd) / stds.T)
        #Restore bias as well
        self.b = ymean - ystd * ((w * means.T) / stds.T).sum()
        break
        
  
  def predict(self, x):
    return x @ self.w + self.b



if __name__=="__main__":
  from sklearn.linear_model import LinearRegression as skl
  
  data = np.load("train-test.npz")
  x_train = data["x_train"]# (700, 1)
  y_train = data["y_train"]# (700,)
  x_test = data["x_test"]# (300, 1)
  y_test = data["y_test"]# (300,)
  
  custom_model = LinearRegression()
  custom_model.fit(array(x_train), array(y_train))
  
  sk_model = skl()
  sk_model.fit(x_train, y_train)
  
  sk_pred = sk_model.predict(x_test)
  custom_pred = custom_model.predict(array(x_test))
  
  print("sklearn      custom")
  for cu, sk in zip(custom_pred[:20], sk_pred[:20]):
    print(sk, cu)
  
  