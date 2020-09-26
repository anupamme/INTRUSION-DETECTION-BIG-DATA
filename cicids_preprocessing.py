import numpy as np
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

InD = np.zeros((0,79),dtype=object)
for x in glob('./MachineLearningCVE/*.csv'):
    print(x)
    InD=np.vstack((InD,pd.read_csv(x)))
    
Dt=InD[:,:-1].astype(float)

#Remove nan values
LNMV=InD[~np.isnan(Dt).any(axis=1),-1]
DtNMV=Dt[~np.isnan(Dt).any(axis=1)]
#Remove Inf values
LNMIV=LNMV[~np.isinf(DtNMV).any(axis=1)]
DtNMIV=DtNMV[~np.isinf(DtNMV).any(axis=1)]

del(DtNMV)

np.save('NBx', MinMaxScaler().fit_transform(DtNMIV))
np.save('NBy', (LNMIV=='BENIGN').astype(int))
np.save('./DistKeras/NBy',(LNMIV=='BENIGN').astype(int).reshape(-1,1))

DtNMIV.shape

MCDt=DtNMIV[LNMIV!='BENIGN',:]

MCDt.shape

MCL=LNMIV[LNMIV!='BENIGN']

LNMIV.shape,MCL.shape

np.save('NMCx', MinMaxScaler().fit_transform(MCDt))
np.save('NMCy', LabelEncoder().fit_transform(MCL))
np.save('./DistKeras/NMCy',OneHotEncoder(sparse=False).fit_transform(MCL.reshape(-1,1)))

#Replace missing values with average and inf values with max
A14=np.average(DtNMIV[:,14])
A15=np.average(DtNMIV[:,15])
M14=np.max(DtNMIV[:,14])
M15=np.max(DtNMIV[:,15])
for c in range(Dt.shape[0]):
    if np.isnan(Dt[c,14]):
        Dt[c,14]=A14
    if np.isnan(Dt[c,15]):
        Dt[c,15]=A15
    if np.isinf(Dt[c,14]):
        Dt[c,14]=M14
    if np.isinf(Dt[c,15]):
        Dt[c,15]=M15
        
np.save('RBx', MinMaxScaler().fit_transform(Dt))
np.save('RBy', (InD[:,-1]=='BENIGN').astype(int))
np.save('./DistKeras/RBy',(InD[:,-1]=='BENIGN').astype(int).reshape(-1,1))

MCDt=Dt[InD[:,-1]!='BENIGN',:]

MCDt.shape

MCL=InD[InD[:,-1]!='BENIGN',-1]

MCL.shape

np.save('RMCx', MinMaxScaler().fit_transform(MCDt))
np.save('RMCy', LabelEncoder().fit_transform(MCL))
np.save('./DistKeras/RMCy',OneHotEncoder(sparse=False).fit_transform(MCL.reshape(-1,1)))