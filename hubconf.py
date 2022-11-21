import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score
# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y = make_blobs(
    n_samples=n_points, n_features=2,
    centers=3,random_state=0
  )
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(
    n_samples=n_points, shuffle=True,
    factor=0.3, noise=0.05, random_state=0
  )
  return X,y
def get_data_mnist():
  digits = load_digits()
  X=digits.data
  y=digits.target
  return X,y

def build_kmeans(X=None,k=10):
   km = KMeans(
    n_clusters=k, random_state=0
  )
   return km

def assign_kmeans(km=None,X=None):
  y_pred = km.fit_predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h = "%.6f"% homogeneity_score(ypred_1, ypred_2)
  c = "%.6f"% completeness_score(ypred_1, ypred_2)
  v = "%.6f"% v_measure_score(ypred_1, ypred_2)
  return h,c,v

###### PART 2 ######

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

def build_lr_model(X=None, y=None):
  # Build logistic regression, refer to sklearn
  lr_model = LogisticRegression(solver="liblinear",fit_intercept=False)
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  # Build Random Forest classifier, refer to sklearn
  rf_model = RandomForestClassifier(random_state=400)
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  y_pred = model.predict(X)
  acc = accuracy_score(y, y_pred)
  prec = precision_score(y, y_pred, average='micro')
  rec =  recall_score(y, y_pred , average='micro')
  f1 =  f1_score(y, y_pred, average='micro' )
  auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr' )
  return acc, prec, rec, f1, auc

from sklearn.model_selection import train_test_split
X, y = get_data_mnist()
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3)

lr_model = build_lr_model(Xtrain, ytrain)
rf_model = build_rf_model(Xtrain, ytrain)

print(get_metrics(lr_model, Xtest, ytest))
print(get_metrics(rf_model, Xtest, ytest))

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  # refer to sklearn documentation on grid search and logistic regression
  lr_param_grid = {
      "max_iter": [100, 200, 500],
      "penalty": ["l1","l2"],
      "solver" : ["liblinear"]
  }
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  # refer to sklearn documentation on grid search and random forest classifier
  rf_param_grid = { 
    'n_estimators' : [1, 10, 100],
    'max_depth' : [1,10,None],
    'criterion' :['gini', 'entropy'],

  }
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  print(model.get_params().keys())
  top1_scores = []

  for scoring in metrics:
    grid_search_cv = GridSearchCV(model,param_grid, cv=cv, scoring=scoring)
    grid_search_cv.fit(X,y)
    top1_scores.append(grid_search_cv.best_score_)

  return top1_scores

param_grid = get_paramgrid_lr()
print("------------")
print(perform_gridsearch_cv_multimetric(model=LogisticRegression(), param_grid=param_grid, cv=5, X=X, y=y, metrics=['accuracy']))

param_grid = get_paramgrid_rf()
print("------------")
print(perform_gridsearch_cv_multimetric(model=RandomForestClassifier(), param_grid=param_grid, cv=5, X=X, y=y, metrics=['accuracy']))

###### PART 3 ######

class MyNN(nn.Module):
  def _init_(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self)
    
    self.fc_encoder = None # write your code inp_dim to hid_dim mapper
    self.fc_decoder = None # write your code hid_dim to inp_dim mapper
    self.fc_classifier = None # write your code to map hid_dim to num_classes
    
    self.relu = None #write your code - relu object
    self.softmax = None #write your code - softmax object
    
  def forward(self,x):
    x = None # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    lc1 = None # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X, y = None, None
  # write your code
  return X,y

def get_loss_on_single_point(mynn=None,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn=None,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimzer.step()
    
  return mynn
