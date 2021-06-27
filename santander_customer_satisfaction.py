#!/usr/bin/env python
# coding: utf-8

# 이 데이터는 산탄데르 은행이 캐글에 경연을 의뢰한 데이터로 피처 이름은 모두 익명 처리되어 이름만으로 속성을 추정하기는 힘들다

# 클래스 테이블 명은 TARGET 이며 이 값이 1이면 불만, 0이면 만족

# 성능 평가는 RCO-AUC로 할 예정 ( 대부분이 만족이므로 정확도 수치보다 더 적합)

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df=pd.read_csv("./train_santander.csv",encoding='latin-1')
print('dataset shape : ',cust_df.shape)
cust_df


# In[4]:


cust_df.info()


# In[5]:


print(cust_df['TARGET'].value_counts())

un=cust_df[cust_df['TARGET']==1].TARGET.count()
total=cust_df.TARGET.count()
print('불만족 비율 : {0:.2f}'.format((un/total)))


# In[6]:


cust_df.describe()


# In[7]:


cust_df['var3'].replace(-999999,2,inplace=True)
cust_df.drop('ID',axis=1,inplace=True)


X_features=cust_df.iloc[:,:-1]
y_labels=cust_df.iloc[:,-1]
print('피처 shape:{0}'.format(X_features.shape))


# In[8]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_features,y_labels,test_size=0.2,random_state=0)

train=y_train.count()
test=y_test.count()

print("학습 shape:{0},테스트 shape:{0}".format(X_train.shape,X_test.shape))

print('학습 세트 레이블 분포 비율')
print(y_train.value_counts()/train)

print('테스트 세트 레이블 분포 비율')
print(y_test.value_counts()/test)


# ### XGBoost

# 사이킷런 래퍼를 이용해 학습을 수행함  
# n_estimators=500으로 설정하고 early_stopping=100으로 설정함/ eval_metric='auc'로 설정하지만 logloss로 설정해도 큰 차이는 없음

# In[9]:


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_clf=XGBClassifier(n_estimators=500,random_state=156)

xgb_clf.fit(X_train,y_train,early_stopping_rounds=100,eval_metric="auc",eval_set=[(X_train,y_train),(X_test,y_test)])

xgb_roc_score=roc_auc_score(y_test,xgb_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))


# In[10]:


from sklearn.model_selection import GridSearchCV

xgb_clf=XGBClassifier(n_estimators=100)

params={'max_depth':[5,7],'min_child_weight':[1,3],'colsample_bytree':[0.5,0.75]}

gridcv=GridSearchCV(xgb_clf,param_grid=params,cv=3)
gridcv.fit(X_train,y_train,early_stopping_rounds=30,eval_metric="auc",
          eval_set=[(X_train,y_train),(X_test,y_test)])

print('GridSearch 최적 파라미터 :',gridcv.best_params_)

xgb_roc_score=roc_auc_score(y_test,gridcv.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC :{0:.4f}'.format(xgb_roc_score))


# colsample_bytree:0.75, max_depth:7, min_child_weight: 1로 설정하고 n_estimators=1000,learning_rate=0.02, reg_alpha=0.03으로 설정한다

# In[12]:


xgb_clf=XGBClassifier(n_estimators=1000,random_state=156,colsample_bytree=0.75, max_depth=7, min_child_weight=1,
                      learning_rate=0.02, reg_alpha=0.03)

xgb_clf.fit(X_train,y_train,early_stopping_rounds=200,eval_metric="auc",eval_set=[(X_train,y_train),(X_test,y_test)])

xgb_roc_score=roc_auc_score(y_test,xgb_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))


# 피처 중요도 그래프

# In[16]:


from xgboost import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig,ax=plt.subplots(1,1,figsize=(10,8))
plot_importance(xgb_clf,ax=ax,max_num_features=20,height=0.4)


# ### LightGBM

# In[20]:


from lightgbm import LGBMClassifier

lgbm_clf=LGBMClassifier(n_estimators=500)
evals=[(X_test,y_test)]

lgbm_clf.fit(X_train,y_train,early_stopping_rounds=100,eval_metric="auc",eval_set=evals,
            verbose=True)

lgbm_roc_score=roc_auc_score(y_test,lgbm_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))


# In[ ]:


from sklearn.model_selection import GridSearchCV

lgbm_clf=LGBMClassifier(n_estimators=100)

params={'num_leaves':[32,64],'max_depth':[128,160],'min_child_samples':[60,100],'subsample':[0.8,1]}

gridcv=GridSearchCV(lgbm_clf,param_grid=params,cv=3)
gridcv.fit(X_train,y_train,early_stopping_rounds=30,eval_metric="auc",
          eval_set=[(X_train,y_train),(X_test,y_test)])

print('GridSearch 최적 파라미터 :',gridcv.best_params_)

xgb_roc_score=roc_auc_score(y_test,gridcv.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC :{0:.4f}'.format(lgbm_roc_score))


# max_depth=128,min_child_samples=100,num_leaves=32,subsample=0.8 하이퍼 파라미터를 적용

# In[21]:


lgbm_clf=LGBMClassifier(n_estimators=500,max_depth=128,min_child_samples=100,num_leaves=32,subsample=0.8 )
evals=[(X_test,y_test)]

lgbm_clf.fit(X_train,y_train,early_stopping_rounds=100,eval_metric="auc",eval_set=evals,
            verbose=True)

lgbm_roc_score=roc_auc_score(y_test,lgbm_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))


# In[ ]:




