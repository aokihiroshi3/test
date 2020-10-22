#%reset -f

ShowYYplots = 1; # Please change to "0" if you do not want to show plots of actural Y and estimated Y
FoldNumber = 3; #Number of fold in cross-validation
FoldNumber_dcv2 = 9; #Number of fold in double cross-validation(入れ子の内側)

import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
from sklearn import svm, tree, model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import contextlib
import math
import supportingfunctions
import warnings

warnings.filterwarnings("ignore")

#Load data set
Originaly, OriginalX, \
 Originaly_prediction1, \
 OriginalX_prediction1, \
 OriginalX_prediction2, \
 data_prediction2pd = supportingfunctions.loadsuperviseddataforregression()
Originaly = np.ravel( Originaly )
Originaly_prediction1 = np.ravel( Originaly_prediction1 )
datapd = pd.read_csv("data.csv", encoding='SHIFT-JIS', index_col=0)
datapd = datapd.ix[:,1:] #Only X
data_prediction1pd = pd.read_csv("data_prediction1.csv", encoding='SHIFT-JIS', index_col=0)
# Delete variables with zero variance
Var0Variable = supportingfunctions.variableszerovariance(OriginalX)
if len(Var0Variable[0]) != 0:
    OriginalX = np.delete(OriginalX, Var0Variable, 1)
    OriginalX_prediction1 = np.delete(OriginalX_prediction1, Var0Variable, 1)
    OriginalX_prediction2 = np.delete(OriginalX_prediction2, Var0Variable, 1)
    datapd = datapd.ix[ :, Var0Variable[0] ]

# Autoscale objective variable (Y) and explanatory variable (X)
X = (OriginalX - OriginalX.mean(axis=0)) / OriginalX.std(axis=0, ddof=1)
y = (Originaly - Originaly.mean(axis=0)) / Originaly.std(axis=0, ddof=1)
X_prediction1 = (OriginalX_prediction1 - OriginalX.mean(axis=0)) / OriginalX.std(axis=0, ddof=1)
X_prediction2 = (OriginalX_prediction2 - OriginalX.mean(axis=0)) / OriginalX.std(axis=0, ddof=1)

Nmin = math.floor( X.shape[0] / FoldNumber )
Nmod = X.shape[0] - Nmin * FoldNumber
Ind = numpy.matlib.repmat( np.arange( 1, FoldNumber+1, 1 ), 1, Nmin).ravel()
if Nmod != 0:
    Ind = np.r_[ Ind, np.arange( 1, Nmod+1, 1 )]
np.random.seed(10000)
FoldIndOut = np.random.permutation(Ind)
np.random.seed()

RegressionMethodNames = ["OLS","PLS","RR","LASSO","EN","LSVR","NLSVR","DT","RF","GP"]
StatisticsNames = ["r2","RMSE","r2cv","RMSEcv","r2pred","RMSEpred","r2dcv","RMSEdcv"]
NumberOfRegressionMethods = len(RegressionMethodNames); #Number of regression methods
LinearRegressionMethodIndex = [ 1, 2, 3, 4, 5, 6 ]
StandardRegressionCoefficientAll = np.zeros( (len(LinearRegressionMethodIndex), X.shape[1]) )
# Calculated and predicted Y
CalculatedYAll = np.empty( (X.shape[0], NumberOfRegressionMethods) )
PredictedYcvAll = np.empty( (X.shape[0], NumberOfRegressionMethods) )
PredictedY1All = np.empty( (X_prediction1.shape[0], NumberOfRegressionMethods) )
PredictedY2All = np.empty( (X_prediction2.shape[0], NumberOfRegressionMethods) )
PredictedYdcvAll = np.empty( (X.shape[0], NumberOfRegressionMethods) )

# 1. Ordinary Least Squares (OLS) or Multivariate Linear Regression (MLR)
# 2. Partial Least Squares (PLS)
# Estimate Y with cross-validation (CV), changing the number of components from 1 to m
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y for each number of components
components = np.arange( 1, np.linalg.matrix_rank(X)+1, 1)
RMSEcvAll = list()
for component in components:
    np.random.seed(10000)
    PlsResult = PLSRegression(n_components=component)
    PredictedYcv = model_selection.cross_val_predict(PlsResult, X, y, cv=FoldNumber)*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
    np.random.seed()
    RMSEcv = supportingfunctions.calc_rmse(Originaly.reshape(len(Originaly),1),PredictedYcv)
    RMSEcvAll.append(RMSEcv)
plt.hold(True);
plt.plot( components, RMSEcvAll, "bo-")
plt.xlabel("# of components for PLS")
plt.ylabel("RMSEcv")
plt.show()
# Decide the optimal number of components with the minimum RMSE value
OptimalComponentNumberPLS = np.where( RMSEcvAll == np.min(RMSEcvAll) )
OptimalComponentNumberPLS = OptimalComponentNumberPLS[0][0]+1
# Construct OLS and PLS model with the optimal number of components and get standard regression coefficient
OlsResult = PLSRegression(n_components=np.linalg.matrix_rank(X))
PlsResult = PLSRegression(n_components=OptimalComponentNumberPLS)
OlsResult.fit(X, y)
PlsResult.fit(X, y)
StandardRegressionCoefficientAll[0,:] = OlsResult.coef_.T
StandardRegressionCoefficientAll[1,:] = PlsResult.coef_.T
# Calculate determinant coefficient and RMSE between actual Y and calculated Y (r2C and RMSEC) and
# determinant coefficient and RMSE between actual Y and estimated Y (r2CV and RMSECV)
CalculatedYAll[:,0] = OlsResult.predict(X).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
CalculatedYAll[:,1] = PlsResult.predict(X).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
np.random.seed(10000)
PredictedYcvAll[:,0] = model_selection.cross_val_predict(OlsResult, X, y, cv=FoldNumber).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
np.random.seed(10000)
PredictedYcvAll[:,1] = model_selection.cross_val_predict(PlsResult, X, y, cv=FoldNumber).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
np.random.seed()
# Prediction
PredictedY1All[:,0] = OlsResult.predict(X_prediction1).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
PredictedY2All[:,0] = OlsResult.predict(X_prediction2).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
PredictedY1All[:,1] = PlsResult.predict(X_prediction1).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)
PredictedY2All[:,1] = PlsResult.predict(X_prediction2).T*Originaly.std(axis=0, ddof=1) + Originaly.mean(axis=0)

# 3. Ridge Regression (RR)
CandidatesOfRidgeLambdas = 2**np.arange( -5, 10, dtype=float) # Candidates of L2 weight
# Estimate objective variable (Y) with cross-validation (CV) for each lambda candidate
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y with CV for each lambda candidate
# Decide the optimal lambda with the minimum RMSE value
RMSECvAll = list()
for CandidateOfRidgeLambdas in CandidatesOfRidgeLambdas:
    RRResult = Ridge(alpha=CandidateOfRidgeLambdas)
    np.random.seed(10000)
    PredictedYcv = model_selection.cross_val_predict(RRResult, X, y, cv=FoldNumber)*Originaly.std(ddof=1) + Originaly.mean()
    np.random.seed()
    RMSECvAll.append(math.sqrt( sum( (Originaly-PredictedYcv)**2 )/ OriginalX.shape[0]))
plt.figure()
plt.plot(CandidatesOfRidgeLambdas, RMSECvAll, 'k', linewidth=2)
plt.xscale("log")
plt.xlabel("Weight for RR")
plt.ylabel("RMSE for RR")
plt.show()

OptimalRRLambda = CandidatesOfRidgeLambdas[np.where( RMSECvAll == np.min(RMSECvAll) )[0][0]]
RRResult = Ridge(alpha=OptimalRRLambda)
RRResult.fit(X, y)
StandardRegressionCoefficientAll[2,:] = RRResult.coef_
# Construct LASSO model with the optimal lambda
CalculatedYAll[:,2] = RRResult.predict(X).T*Originaly.std(ddof=1) + Originaly.mean()
np.random.seed(10000)
PredictedYcvAll[:,2] = model_selection.cross_val_predict(RRResult, X, y, cv=FoldNumber).T*Originaly.std(ddof=1) + Originaly.mean()
np.random.seed()
# Prediction
PredictedY1All[:,2] = RRResult.predict(X_prediction1).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedY2All[:,2] = RRResult.predict(X_prediction2).T*Originaly.std(ddof=1) + Originaly.mean()

# 4. Least Absolute Shrinkage and Selection Operator (LASSO)
CandidatesOfLASSOLambda = np.arange(0.01, 0.71, 0.01, dtype=float) # Candidates of L1 weight
# Estimate objective variable (Y) with cross-validation (CV) for each lambda candidate
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y with CV for each lambda candidate
# Decide the optimal lambda with the minimum RMSE value
np.random.seed(10000)
LASSOResult = LassoCV(cv=FoldNumber, alphas=CandidatesOfLASSOLambda)
LASSOResult.fit(X, y)
np.random.seed()
plt.figure()
plt.plot(LASSOResult.alphas_, LASSOResult.mse_path_.mean(axis=-1), 'k', linewidth=2)
plt.xlabel("Lambda for LASSO")
plt.ylabel("MSE for LASSO")
plt.show()
OptimalLASSOLambda = LASSOResult.alpha_
LASSOResult = Lasso(alpha=OptimalLASSOLambda)
LASSOResult.fit(X, y)
StandardRegressionCoefficientAll[3,:] = LASSOResult.coef_.T
# Construct LASSO model with the optimal lambda
CalculatedYAll[:,3] = LASSOResult.predict(X).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedYcvAll[:,3] = model_selection.cross_val_predict(LASSOResult, X, y, cv=FoldNumber).T*Originaly.std(ddof=1) + Originaly.mean()
# Prediction
PredictedY1All[:,3] = LASSOResult.predict(X_prediction1).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedY2All[:,3] = LASSOResult.predict(X_prediction2).T*Originaly.std(ddof=1) + Originaly.mean()

# 5. Elastic Net (EN)
CandidatesOfENLambda = np.arange(0.01, 0.71, 0.01, dtype=float) # Candidates of lambda
CandidatesOfENAlpha = np.arange(0.01, 1.00, 0.01, dtype=float) # Candidates of alpha
# Estimate objective variable (Y) with cross-validation (CV) for each lambda candidate and eachalpha candidate
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y with CV for each lambda candidate and eachalpha candidate
# Decide the optimal lambda with the minimum RMSE value
np.random.seed(10000)
ENResult = ElasticNetCV(cv=FoldNumber, l1_ratio=CandidatesOfENAlpha, alphas=CandidatesOfENLambda)
ENResult.fit(X, y)
np.random.seed()
OptimalAlphaNumberEN = np.where( CandidatesOfENAlpha == ENResult.l1_ratio_ )
OptimalAlphaNumberEN = OptimalAlphaNumberEN[0][0]
MSEaverageEN = ENResult.mse_path_.mean(axis=-1)
plt.figure()
plt.plot(ENResult.alphas_, MSEaverageEN[OptimalAlphaNumberEN,:], 'k', linewidth=2)
plt.xlabel("Lambda for EN")
plt.ylabel("MSE for EN")
plt.show()
OptimalENLambda = ENResult.alpha_
OptimalENAlpha = ENResult.l1_ratio_
ENResult = ElasticNet(l1_ratio=OptimalENAlpha, alpha=OptimalENLambda)
ENResult.fit(X, y)
StandardRegressionCoefficientAll[4,:] = ENResult.coef_.T
# Construct LASSO model with the optimal lambda
CalculatedYAll[:,4] = ENResult.predict(X).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedYcvAll[:,4] = model_selection.cross_val_predict(ENResult, X, y, cv=FoldNumber).T*Originaly.std(ddof=1) + Originaly.mean()
# Prediction
PredictedY1All[:,4] = ENResult.predict(X_prediction1).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedY2All[:,4] = ENResult.predict(X_prediction2).T*Originaly.std(ddof=1) + Originaly.mean()

# 6. Linear Support Vector Regression (LSVR)
CandidatesOfLSVRC = 2**np.arange( -5, 5, dtype=float) # Candidates of C
CandidatesOfLSVREpsilon = 2**np.arange( -10, 0, dtype=float) # Candidates of epsilon
# Estimate objective variable (Y) with cross-validation (CV) for each combination of C and epsilon candidates
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y for each combination of C and epsilon candidates
# Decide the optimal combination of C and epsilon with the minimum RMSE value
TunedParameters = {'C':CandidatesOfLSVRC, 'epsilon':CandidatesOfLSVREpsilon}
np.random.seed(10000)
LSVRResltGS = GridSearchCV(svm.SVR(kernel='linear'), TunedParameters, cv=FoldNumber,scoring='neg_mean_squared_error')
LSVRResltGS.fit(X, y)
np.random.seed()
OptimalLSVRC = LSVRResltGS.best_params_['C']
OptimalLSVREpsilon = LSVRResltGS.best_params_['epsilon']
# Construct SVR model with the optimal C, epsilon and gamma
LSVRResult = svm.SVR(kernel='linear', C=OptimalLSVRC, epsilon=OptimalLSVREpsilon)
LSVRResult.fit(X, y)
CalculatedYAll[:,5] = LSVRResult.predict(X).T*Originaly.std(ddof=1) + Originaly.mean()
StandardRegressionCoefficientAll[5,:] = LSVRResult.coef_
np.random.seed(10000)
PredictedYcvAll[:,5] = model_selection.cross_val_predict(LSVRResult, X, y, cv=FoldNumber).T*Originaly.std(ddof=1) + Originaly.mean()
np.random.seed()
# Prediction
PredictedY1All[:,5] = LSVRResult.predict(X_prediction1).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedY2All[:,5] = LSVRResult.predict(X_prediction2).T*Originaly.std(ddof=1) + Originaly.mean()

# 7. Non-Linear Support Vector Regression (NLSVR)
CandidatesOfNLSVRC = 2**np.arange( -5, 10, dtype=float) # Candidates of C
CandidatesOfNLSVREpsilon = 2**np.arange( -10, 0, dtype=float) # Candidates of epsilon
CandidatesOfNLSVRGamma = 2**np.arange( -20, 10, dtype=float) # Candidates of gamma
# Calculate gram matrix of Gaussian kernel and its variance for each gamma candidate
# Decide the optimal gamma with the maximum variance value
OptimalNLSVRGamma = supportingfunctions.optimize_gamma_grammatrix(X, CandidatesOfNLSVRGamma)
# Estimate objective variable (Y) with cross-validation (CV) for each combination of C and epsilon candidates
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y for each combination of C and epsilon candidates
# Decide the optimal combination of C and epsilon with the minimum RMSE value
TunedParameters = {'C':CandidatesOfNLSVRC, 'epsilon':CandidatesOfNLSVREpsilon}
np.random.seed(10000)
NLSVRResltGS = GridSearchCV(svm.SVR(kernel='rbf', gamma=OptimalNLSVRGamma), TunedParameters, cv=FoldNumber,scoring='neg_mean_squared_error' )
NLSVRResltGS.fit(X, y)
np.random.seed()
OptimalNLSVRC = NLSVRResltGS.best_params_['C']
OptimalNLSVREpsilon = NLSVRResltGS.best_params_['epsilon']
# Construct SVR model with the optimal C, epsilon and gamma
NLSVRResult = svm.SVR(kernel='rbf', C=OptimalNLSVRC, epsilon=OptimalNLSVREpsilon, gamma=OptimalNLSVRGamma)
NLSVRResult.fit(X, y)
CalculatedYAll[:,6] = NLSVRResult.predict(X).T*Originaly.std(ddof=1) + Originaly.mean()
np.random.seed(10000)
PredictedYcvAll[:,6] = model_selection.cross_val_predict(NLSVRResult, X, y, cv=FoldNumber).T*Originaly.std(ddof=1) + Originaly.mean()
np.random.seed()
# Prediction
PredictedY1All[:,6] = NLSVRResult.predict(X_prediction1).T*Originaly.std(ddof=1) + Originaly.mean()
PredictedY2All[:,6] = NLSVRResult.predict(X_prediction2).T*Originaly.std(ddof=1) + Originaly.mean()

# 8. Decision Tree (DT)
MaxCandidateOfMaxDepthDT = 100 #Maximum depth of tree
MinSamplesLeafDT = 3 #Minimum number of samples for one leaf
CandidatesOfDTDepth = np.arange( 1, MaxCandidateOfMaxDepthDT+1, 1)
# Construct DT and Prune DT
RMSECvAll = list()
for CandidateOfMaxDepth in CandidatesOfDTDepth:
    DTResult = tree.DecisionTreeRegressor(max_depth=CandidateOfMaxDepth, min_samples_leaf=MinSamplesLeafDT)
    np.random.seed(10000)
    PredictedYcv = model_selection.cross_val_predict(DTResult, OriginalX, Originaly, cv=FoldNumber)
    np.random.seed()
    RMSECvAll.append(math.sqrt( sum( (Originaly-PredictedYcv)**2 )/ OriginalX.shape[0]))
plt.figure()
plt.plot(CandidatesOfDTDepth, RMSECvAll, 'k', linewidth=2)
plt.xlabel("Depth of tree for DT")
plt.ylabel("RMSE in CV for DT")
plt.show()
OptimalMaxDepthDT = CandidatesOfDTDepth[np.where( RMSECvAll == np.min(RMSECvAll) )[0][0] ]
DTResult = tree.DecisionTreeRegressor(max_depth=OptimalMaxDepthDT, min_samples_leaf=MinSamplesLeafDT)
DTResult.fit( OriginalX, Originaly )
CalculatedYAll[:,7] = DTResult.predict(OriginalX)
np.random.seed(10000)
PredictedYcvAll[:,7] = model_selection.cross_val_predict(DTResult, OriginalX, Originaly, cv=FoldNumber)
np.random.seed()
# Check rules of DT
datapdDT = pd.read_csv("data.csv", encoding='SHIFT-JIS', index_col=0)
with contextlib.closing(StringIO()) as DTfile:
    tree.export_graphviz(DTResult, out_file=DTfile,
                         feature_names=datapdDT.columns[1:],
                         class_names=datapdDT.columns[0])
    output = DTfile.getvalue().splitlines()
output.insert(1, 'node[fontname="meiryo"];')
with open('DTResult.dot', 'w') as f:
    f.write('\n'.join(output))
# Estimate Y for new samples based on DT in 1. and 2.
PredictedY1All[:,7] = DTResult.predict(OriginalX_prediction1)
PredictedY2All[:,7] = DTResult.predict(OriginalX_prediction2)

# 9. Random Forest (RF)
NumberOfTreesRF = 500 # 1. Number of decision trees
CandidatesOfXvariablesRateRF = np.arange( 1, 10, dtype=float)/10 #candidates of the ratio of the number of explanatory variables (X) for decision trees
# Run RFR for every candidate of X-ratio and estimate values of objective variable (Y) for Out Of Bag (OOB) samples
# Calculate Root-Mean-Squared Error (RMSE) between actual Y and estimated Y for each candidate of X-ratio
RMSEoobAll = list()
for CandidateOfXvariablesRate in CandidatesOfXvariablesRateRF:
    np.random.seed(10000)
    RandomForestResult = RandomForestRegressor(n_estimators=NumberOfTreesRF, max_features=int(max(math.ceil(OriginalX.shape[1]*CandidateOfXvariablesRate),1)), oob_score=True)
    RandomForestResult.fit(OriginalX, Originaly)
    np.random.seed()
    RMSEoobAll.append(math.sqrt( sum( (Originaly-RandomForestResult.oob_prediction_)**2 )/ OriginalX.shape[0]))
plt.figure()
plt.plot(CandidatesOfXvariablesRateRF, RMSEoobAll, 'k', linewidth=2)
plt.xlabel("Ratio of the number of X-variables for RF")
plt.ylabel("RMSE of OOB for RF")
plt.show()
# Decide the optimal X-ratio with the minimum RMSE value
OptimalXvariablesRateRF = CandidatesOfXvariablesRateRF[ np.where( RMSEoobAll == np.min(RMSEoobAll) )[0][0] ]
# Construct RFR model with the optimal X-ratio
np.random.seed(10000)
RandomForestResult = RandomForestRegressor(n_estimators=NumberOfTreesRF, max_features=int(max(math.ceil(OriginalX.shape[1]*OptimalXvariablesRateRF),1)), oob_score=True)
RandomForestResult.fit(OriginalX, Originaly)
np.random.seed()
CalculatedYAll[:,8] = RandomForestResult.predict(OriginalX)
np.random.seed(10000)
PredictedYcvAll[:,8] = model_selection.cross_val_predict(RandomForestResult, OriginalX, Originaly, cv=FoldNumber)
np.random.seed()
# Estimate Y based on the RFR model in 6.
PredictedY1All[:,8] = RandomForestResult.predict(OriginalX_prediction1)
PredictedY2All[:,8] = RandomForestResult.predict(OriginalX_prediction2)

# 10. Gaussian Process (GP)
# Construct GP model
np.random.seed(10000)
GPResults = GaussianProcessRegressor(kernel=Matern())
#GPResults = GaussianProcessRegressor(kernel=RBF())
GPResults.fit(X, Originaly)
np.random.seed()
CalculatedYAll[:,9] = GPResults.predict(X)
np.random.seed(10000)
PredictedYcvAll[:,9] = model_selection.cross_val_predict(GPResults, X, Originaly, cv=FoldNumber)
np.random.seed()
# Prediction
PredictedY1All[:,9] = GPResults.predict(X_prediction1)
PredictedY2All[:,9] = GPResults.predict(X_prediction2)

# Calculate statistics and show YY plots
YMax = np.max( np.array([np.max(Originaly), np.max(CalculatedYAll), np.max(PredictedYcv)]))
YMin = np.min( np.array([np.min(Originaly), np.min(CalculatedYAll), np.min(PredictedYcv)]))
YMaxPred = np.max( np.array([np.max(Originaly_prediction1),np.max(PredictedY1All)]))
YMinPred = np.min( np.array([np.min(Originaly_prediction1),np.min(PredictedY1All)]))
Statistics = np.zeros( (6, NumberOfRegressionMethods) )
for RegressionMethodNumber in range(0,NumberOfRegressionMethods):
    Statistics[0,RegressionMethodNumber] = supportingfunctions.calc_r2(Originaly, CalculatedYAll[:,RegressionMethodNumber])
    Statistics[1,RegressionMethodNumber] = supportingfunctions.calc_rmse(Originaly, CalculatedYAll[:,RegressionMethodNumber])
    Statistics[2,RegressionMethodNumber] = supportingfunctions.calc_r2(Originaly, PredictedYcvAll[:,RegressionMethodNumber])
    Statistics[3,RegressionMethodNumber] = supportingfunctions.calc_rmse(Originaly, PredictedYcvAll[:,RegressionMethodNumber])
    Statistics[4,RegressionMethodNumber] = supportingfunctions.calc_r2(Originaly_prediction1, PredictedY1All[:,RegressionMethodNumber])
    Statistics[5,RegressionMethodNumber] = supportingfunctions.calc_rmse(Originaly_prediction1, PredictedY1All[:,RegressionMethodNumber])
    if ShowYYplots != 0:
        supportingfunctions.make_yyplot( Originaly, CalculatedYAll[:,RegressionMethodNumber], YMax, YMin, "Calculated Y" + " " + RegressionMethodNames[RegressionMethodNumber])
        supportingfunctions.make_yyplot( Originaly, PredictedYcvAll[:,RegressionMethodNumber], YMax, YMin, "Y Estimated with cross-validation" + " " + RegressionMethodNames[RegressionMethodNumber])
        supportingfunctions.make_yyplot( Originaly_prediction1, PredictedY1All[:,RegressionMethodNumber], YMaxPred, YMinPred, "Estimated Y(test)" + " " + RegressionMethodNames[RegressionMethodNumber])

# Show results
print( "....................................." )
print( "Optimized hyperparameter-values" )
print( "....................................." )
print("Optimal number of PLS components: {0}".format(OptimalComponentNumberPLS))
print("Optimal RR weight: {0}".format(OptimalRRLambda))
print("Optimal LASSO lambda: {0}".format(OptimalLASSOLambda))
print("Optimal EN lambda: {0}".format(OptimalENLambda))
print("Optimal EN alpha: {0}".format(OptimalENAlpha))
print("Optimal LSVR C: {0}".format(OptimalLSVRC))
print("Optimal LSVR Epsilon: {0}".format(OptimalLSVREpsilon))
print("Optimal NLSVR C: {0}".format(OptimalNLSVRC))
print("Optimal NLSVR Epsilon: {0}".format(OptimalNLSVREpsilon))
print("Optimal NLSVR Gamma: {0}".format(OptimalNLSVRGamma))
print("Optimal depth of tree for DT: {0}".format(OptimalMaxDepthDT))
print("Optimal ratio of the number of X-variables for RF: {0}".format(OptimalXvariablesRateRF))

# Save results
CalculatedYAll = pd.DataFrame(CalculatedYAll)
CalculatedYAll.index = datapd.index
CalculatedYAll.columns = RegressionMethodNames
CalculatedYAll.to_csv( "CalculatedY.csv" )
PredictedYcvAll = pd.DataFrame(PredictedYcvAll)
PredictedYcvAll.index = datapd.index
PredictedYcvAll.columns = RegressionMethodNames
PredictedYcvAll.to_csv( "PredictedYcv.csv" )
PredictedY1All = pd.DataFrame(PredictedY1All)
PredictedY1All.index = data_prediction1pd.index
PredictedY1All.columns = RegressionMethodNames
PredictedY1All.to_csv( "PredictedY1.csv" )
PredictedY2All = pd.DataFrame(PredictedY2All)
PredictedY2All.index = data_prediction2pd.index
PredictedY2All.columns = RegressionMethodNames
PredictedY2All.to_csv( "PredictedY2.csv" )
StandardRegressionCoefficientAll = pd.DataFrame(StandardRegressionCoefficientAll)
StandardRegressionCoefficientAll.columns = datapd.columns
StandardRegressionCoefficientAll.index = RegressionMethodNames[:6]
StandardRegressionCoefficientAll.to_csv( "StandardRegressionCoefficients.csv", encoding="shift-jis" )
Statistics = pd.DataFrame(Statistics)
Statistics.columns = RegressionMethodNames
Statistics.index = StatisticsNames[:6]
Statistics.to_csv( "StatisticAll.csv" )
