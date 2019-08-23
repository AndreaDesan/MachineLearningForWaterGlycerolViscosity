import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats
def anova_analysis(X,y,predictions,beta,alpha=0.05):
    '''
    The function performs and repors the ANOVA analysis for a given SKLearn multivariable linear regression
    It also gives the individual p-values for the regression coefficients beta_j
    
    The arguments of the function are:
    X: predictor matrix (it is assumed that it is passed as a Pandas DataFrame)
    y: response vector (it is assumed to be a Pandas Series)
    beta: coefficients vector
    alpha: significance level (having a default value of 0.05)
    '''
    Xext=X.values #convert X to a numpy array
    #Xext=X #uncomment this line and comment the line above is X is passed a numpy ndarray
    
    #evaluates number of samples (n) and number of predictors (p)
    n,p=Xext.shape 
    
    #modify X in such a way that the linear regression can be written as y=X*beta+resid in matix form
    Xext = np.append(np.ones((n,1)), X, axis=1) 
    
    res=y-predictions #calculate residuals
    
    rss=np.dot(res.T,res) #Residuals Sum of Squares
    
    rse=np.sqrt(rss/(n-p-1)) #Residual Standard Error
    
    #calculate the variance-covariance matrix of beta C=RSE^2*(X*X.T)^-1
    c=rse**2*(np.linalg.inv(np.dot(Xext.T,Xext)))
    
    se_beta=np.sqrt(c.diagonal()) #standard error of beta
    
    t_beta=beta/se_beta #beta t-statistic
    
    #calculated p-values of the beta coefficients
    p_values = np.array([2*(1-stats.t.cdf(np.abs(i),(n-p-1))) for i in t_beta])
    
    tss=sum((y-y.mean())**2) #Total Sum of Squares
    
    f_stat=(tss-rss)/p/((rss)/(n-p-1)) #calculate the F-statistic for the linear regression
    
    #evaluate the F-disribution with dimensions p,n-p-1 and the critical f value for a significance equal to alpha
    fdistribution = stats.f(p,n-p-1)
    f_critic=fdistribution.ppf(1-alpha)
    
    #calculate the p-value of f_stat
    f_p_value=1-fdistribution.cdf(f_stat)
      
    #create and print a dataframe summarising the ANOVA results
    f_summary=pd.DataFrame()
    f_summary['Source of Var']=['Regression','Error','Total']
    f_summary['DoF']=[p,n-p-1,n-1] #degrees of freedom
    f_summary['SS']=[sum((predictions-y.mean())**2) ,rss, tss] #sum of squares
    f_summary['MS']=[round(sum((predictions-y.mean())**2)/p,2),round(rss/(n-p-1),2),''] #Mean squares
    f_summary['F-stat']=[round(f_stat,2),'','']
    f_summary['F-crit']=[round(f_critic,2),'','']
    f_summary['p-val']=[round(f_p_value,5),'','']
    f_summary['H0 hypothesis']=[f_p_value > alpha,'','']
    f_summary['R2']=[round(metrics.explained_variance_score(y, predictions),5),'','']
    print(f_summary.round(2))
    
    #create a print a dataframe containing the individual p-values for the beta coefficients
    beta_summary=pd.DataFrame()
    beta_summary['Predictor']=X.columns.insert(0,'intercept')
    beta_summary['Coeff values']=beta.round(2) #regression coefficients
    beta_summary['SE']=se_beta.round(2) #standard error
    beta_summary['T-stat']=t_beta.round(2) #t-statistic
    beta_summary['p-value']=p_values.round(5)
    beta_summary['H0 hypothesis']=p_values > alpha
    print(beta_summary)