###---------------------------------------Problem Statement---------------------------------------###
###'''Imagine you are loooking for a e-comers company And the marketing team target to coastomer more effectly,
### by creating segments Based on there purchasing behaiviour, how ever,
### traaditional segmentatiion methods are not providing satisfaction results''''
###fulfrom of DBSCAN density based spatial clustring of application of noise
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
np.random.seed(42)
data= np.random.rand(100,2)
print(data)
scaller= StandardScaler()
data_scaled= scaller.fit_transform(data)
dbscan= DBSCAN(eps=0.3,min_samples=5)
cluster= dbscan.fit_predict(data_scaled)
plt.scatter(data[:,0],data[:,1],c=cluster)
plt.title('DBCAN Clusytering')
plt.xlabel('Feature-1')
plt.ylabel('Feature-2')
plt.legend()
plt.show()
result_df= pd.DataFrame({'feature-1':data[:,0],
                         'feature-2':data[:,1],
                         'cluster':cluster})
print('number of cluster: ',len(np.unique(cluster)))
print('size of each cluster: ')
print(result_df['cluster'].value_counts())
use_input= np.array([[0.6,0.7]])
user_input_sclaed= scaller.transform(use_input)
user_cluster= dbscan.fit_predict(user_input_sclaed)
print(f'the user input belongs to cluster: {user_cluster[0]}')