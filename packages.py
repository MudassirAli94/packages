#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# FUnction to understand variable importances using Shap
def shap_imp(model_obj,test_df,var_cat,exclude_vars = None):
    # Top contributors from training data
    shap_values_f = shap.TreeExplainer(model_obj).shap_values(test_df)
    
    #Get the shapley values of training data to understand contributions
    shap_v_f = pd.DataFrame(shap_values_f)
    #Rename columns names
    shap_v_f.columns = test_df.columns
    # Get the absolute values
    shap_abs_f = np.abs(shap_v_f)
    # Get the overall score by taking average absolute shap score
    var_imp_f = pd.DataFrame(shap_abs_f.mean().reset_index())
    #rename column names
    var_imp_f.columns = ['Variable' , 'Mean_Abs_SHAP']
    
    # Get the overall shap score across all variables to get percentage contribution of each variable
    sum_shap_f = var_imp_f.Mean_Abs_SHAP.sum()
    # Get percentage importance and sort
    var_imp_f['Pct_Importance'] = var_imp_f['Mean_Abs_SHAP']/sum_shap_f
    var_imp_f = var_imp_f.sort_values(['Pct_Importance'], ascending = False)
    
    #Join by variable categories
    var_imp_f = pd.merge(var_imp_f,var_cat, how = 'left', left_on = 'Variable', right_on = 'column_names')
    
    # Summaraize importances at different levels of variable categories
    categ1_imp_f = var_imp_f.groupby('categ_level1').agg(
        pct_imp = pd.NamedAgg(column = 'Pct_Importance' , aggfunc = 'sum')
    ).reset_index().sort_values('pct_imp', ascending = False)
    
    categ1_2_3_imp_f = var_imp_f.groupby(['categ_level1' , 'Variable']).agg(
        pct_imp = pd.NamedAgg(column = "Pct_Importance" , aggfunc = 'sum')
    ).reset_index().sort_values('pct_imp' , ascending = False)
    
    # Add in correlation sign
    df_abs_shap = ABS_SHAP(shap_values_f , test_df, exclude_vars)
    df_abs_shap['Sign'] = np.where(df_abs_shap['Corr'] > 0, 'Positive Correlation' , 'Negative Correlation')
    df_abs_shap['Sign'] = np.where(df_abs_shap['Corr'] == 0 , 'No Correlation' , df_abs_shap['Sign'])
    categ1_2_3_imp_f = pd.merge(categ1_2_3_imp_f, df_abs_shap, how = 'left' , on = 'Variable')
    
    #Add in overall category level importances
    categ1_imp1_f = categ1_imp_f
    categ1_imp1_f = categ1_imp1_f.rename(columns =  {'pct_imp' : 'category_lvl_pct_imp'})
    categ1_2_3_imp_f = pd.merge(categ1_2_3_imp_f, categ1_imp1_f, how = 'left', on = 'categ_level1')
    categ1_2_3_imp_f = categ1_2_3_imp_f.sort_values(['category_lvl_pct_imp', 'pct_imp'], ascending = False)
    categ1_2_3_imp_f = categ1_2_3_imp_f.drop(['SHAP_abs'],axis = 1)
    
    return categ1_imp_f,categ1_2_3_imp_f

def ABS_SHAP(df_shap,df,exclude_vars = None):
    # Make a copy of input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index()
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        index_exc = ~df_v[i].isnull()
        b = np.corrcoef(shap_v[index_exc][i], df_v[index_exc][i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis = 1).fillna(0)
    # Make a data frame. Col 1 is the feature and Col 2 is the correlation coefficient
    corr_df.columns = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0 , 'blue' , 'red')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable' , 'SHAP_abs']
    k2 = k.merge(corr_df, left_on = 'Variable', right_on = 'Variable', how = 'inner')
    k2 = k2.sort_values(by = 'SHAP_abs' , ascending = True)
    
    if exclude_vars:
        k2 = k2[~(k2['Variable'].str.contains(exclude_vars))]
        
    colorlist = k2['Sign']
    ax = k2.plot.barh(x = 'Variable' , y = 'SHAP_abs' , color = colorlist , figsize = (20,50) , 
                      legend = False)
    ax.set_xlabel("SHAP Value (Purple = Positive Impact)", fontsize = 30)
    ax.set_ylabel("Variables", fontsize = 30)
    plt.title('Variable Importance', fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    

    return k2.sort_values(by = 'SHAP_abs' , ascending = False)

def lift_plot(df_lift,x,y,ngroup,weight = None):
    df_ = df_lift.copy()
    
    # Get the proportion of data to divide into groups
    probs = np.linspace(0,1,(ngroup + 1))
    
    # If weight is not available, make it 1
    if weight is None:
        df_['weight'] = 1.0
    else:
        df_['weight'] = df_[weight].values
        
    # Get the cut-off points to divide data into 'ngroup' using quantile function
    x_wq = [wq.quantile(df_[x], quantile = prob, weights = df_['weight']) for prob in probs]
    x_wq = list(dict.fromkeys(x_wq))
    
    # Get the count of groups
    ngroup_dedup = len(x_wq) - 1
    # Get the labels from 1 to 'ngroup'
    labels = np.linsapce(1,mgroup_dedup,ngroup_dedup).astype(str)
    
    # Label the data into ngroups based on probability scores
    df_['bin'] = pd.cut(df_[x], bix = x_wq, labels = labels, include_lowest = True , duplicates = 'drop')
    df_['bin'] = np.where(df_['bin'].isnull() , 'missing' , df_['bin'])
    
    # To calculate weighted score
    df_['y_true'] = df_[y] * df_['weight']
    
    # Get the total sum metrics
    sum_y_true = np.sum(df_['y_true'])
    sum_weight = np.sum(df_['weight'])
    
    #overall average
    
    avg_y_true = sum_y_true/sum_weight
    
    total_weight = df_['weight'].sum()
    total_true = df_['y_true'].sum()
    
    # Calculate overall metrics of min,max probablity, total records, pct true prediction captured for each group
    
    df_agg_lift = df_.groupby('bin').agg(
    x_min = pd.NamedAgg(column = x, aggfunc = 'min'),
    x_max = pd.NamedAgg(column = x, aggfunc = 'max'),
    num_records = pd.NamedAgg(column = x, aggfunc = 'size'),
    sum_y_true = pd.NamedAgg(column = 'y_true', aggfunc = 'sum'),
    sum_weight = pd.NamedAgg(column = 'weight', aggfunc = 'sum')
    ).assign(
        pct_y_true = lambda dat:dat['sum_y_true']/dat['sum_weight']).reset_index()
    
    prinmt("Overall True Average Prob" , avg_y_true)
    df_agg_lift['rel_y_true'] = df_agg_lift['pct_y_true']/avg_y_true
    df_agg_lift['pct_all_conversions'] = df_agg_lift['sum_y_true']/total_true
    
    return df_,df_agg_lift

