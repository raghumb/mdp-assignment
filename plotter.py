import matplotlib
matplotlib.use('Agg') 			  		 			 	 	 		 		 	  		   	  			  	
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_curve_single(x_values, title, x_title, y_title, prefix = None):
    plt.clf()
    #index = title.find('Learner')
    #title = title[:index]
    label = y_title + ' Vs ' + x_title
    plt.plot(x_values)
    plt.ylabel(y_title, fontsize = 14)
    plt.xlabel(x_title, fontsize = 14)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    #plt.ylim(0,1)
    #plt.yticks(np.arange(0, 1, 0.1))
    
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/' + title + '.png')
    plt.clf()


def plot_value_fn(df, title):
    #fig, ax = plt.subplots(2,2, figsize=(20,10))
    
    sns_plot  = sns.lineplot(data=df, x='gamma', y='value', hue='state')
    sns_plot.set_title(title)  
    fig = sns_plot.get_figure()
    fig.savefig('images/' + title + '.png')

    

def plot_rewards(episode_rewards, title, x_title, y_title, prefix = None):
    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    smoothing_window=10
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/' + title + '.png')
    plt.clf()

def plot_curve(x_values, y_values, title, x_title, y_title, prefix = None):
    
    #index = title.find('Learner')
    #title = title[:index]
    label = y_title + ' Vs ' + x_title
    plt.plot(x_values, y_values, label = label)
    plt.ylabel(y_title, fontsize = 14)
    plt.xlabel(x_title, fontsize = 14)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    #plt.ylim(0,1)
    #plt.yticks(np.arange(0, 1, 0.1))
    
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/' + title + '.png')
    plt.clf()

def plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, title, prefix = None):
    
    
    plt.plot(train_sizes, train_scores_mean, label = 'Training Accuracy')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation Accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for ' + title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.title('Learning Curve for ' + title)
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/learning_curve - ' + title + '.png')
    plt.clf()


def plot_model_complexity_curve_old(param_name, param_range, train_mean, train_std, validation_mean, validation_std, title, prefix = None):
    plt.plot(param_range, train_mean, color='blue', label='training score')
    plt.plot(param_range, validation_mean, color='red', label='validation score')
    #plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    #plt.fill_between(param_range, validation_mean - validation_std, validation_mean + validation_std, color="gainsboro")
    plt.legend(loc='best')
    t = train_mean + train_std
    v = validation_mean + validation_std
    m1 = max(t)
    m2 = max(v) 
    max_val = max(m1, m2)
    print(max_val)
    t = train_mean - train_std
    v = validation_mean - validation_std
    m1 = min(t)
    m2 = min(v)   
    min_val = min(m1, m2)  
    print(min_val)

    tick_range = (max_val - min_val )/.2
    print('max val ' + str(max_val))
    print('min val ' + str(min_val))
    print('tick range ' + str(tick_range))
    plt.ylim(min_val, max_val)
    plt.yticks(np.arange(min_val, max_val, 0.01))
    plt.xlabel(param_name)
    plt.ylabel('score')  
    #plt.yscale('symlog', linthreshx = 5.)
    #ax1 = plt.subplot(212)
    #ax1.set_yscale('log')#
    #plt.yscale('linear')
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                wspace=0.35)
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/Model Complexity Curve - '+ title + ' - ' + param_name + '.png')
    plt.clf()


def plot_model_complexity_curve(param_name, param_range, train_mean, train_std, validation_mean, validation_std, title, prefix = None,
                                title_pre = None):

    plt.plot(param_range, train_mean, color='blue', label='Training Accuracy')
    plt.plot(param_range, validation_mean, color='red', label='Validation Accuracy')
    #plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    #plt.fill_between(param_range, validation_mean - validation_std, validation_mean + validation_std, color="gainsboro")
    plt.legend(loc='best')
    t = train_mean + train_std
    v = validation_mean + validation_std
    m1 = max(t)
    m2 = max(v) 
    max_val = max(m1, m2)
    print(max_val)
    t = train_mean - train_std
    v = validation_mean - validation_std
    m1 = min(t)
    m2 = min(v)   
    min_val = min(m1, m2)  
    print(min_val)

    
    print('max val ' + str(max_val))
    print('min val ' + str(min_val))    
    plt.ylim(min_val, max_val)
    
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt_title = title
    if title_pre is not None:
        plt_title =  plt_title + ' ' +  title_pre
    plt.title('Model Complexity for ' + plt_title, fontsize = 12)
    plt.grid(True)
    #plt.figure(figsize=(50,50))  
    
    #plt.yscale('symlog', linthreshx = 5.)
    #ax1 = plt.subplot(212)
    #ax1.set_yscale('log')#
    #plt.yscale('linear')
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                wspace=0.35)
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/Model Complexity Curve - '+ title + ' - ' + param_name + '.png')
    plt.clf()        

def plot_scatter(x, y, xlabel, ylabel, title):
    plt.plot(x, y, 'o', color='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.figure(figsize=(2,2))  
    plt.savefig('images/Scatter Plot - '+ title + '.png')
    plt.tight_layout()    
    plt.clf()

def test():
    for i in range (1, 3):
        plt.subplot(2, 1, i)
        
        param_name = 'pname'
        param_range = np.array([1,2,3,4,5,6])
        train_mean = np.array([1,2,3,4,5,6])
        train_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        validation_mean = np.array([11,12,13,14,15,16])
        validation_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        title = 'test title1Learner'
        prefix = None
        index = title.find('Learner')
        title = title[:index]
        plt.plot(param_range, train_mean, color='blue', label='Training Accuracy')
        plt.plot(param_range, validation_mean, color='red', label='Validation Accuracy')
        #plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
        #plt.fill_between(param_range, validation_mean - validation_std, validation_mean + validation_std, color="gainsboro")
        plt.legend(loc='best')
        t = train_mean + train_std
        v = validation_mean + validation_std
        m1 = max(t)
        m2 = max(v) 
        max_val = max(m1, m2)
        print(max_val)
        t = train_mean - train_std
        v = validation_mean - validation_std
        m1 = min(t)
        m2 = min(v)   
        min_val = min(m1, m2)  
        print(min_val)

        
        print('max val ' + str(max_val))
        print('min val ' + str(min_val))    
        plt.ylim(min_val, max_val)
        
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')
        plt.title(title)
        #plt.figure(figsize=(2,2))  
        plt.grid(True)
        plt.tight_layout() 
        
    #plt.yscale('symlog', linthreshx = 5.)
    #ax1 = plt.subplot(212)
    #ax1.set_yscale('log')#
    #plt.yscale('linear')
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                wspace=0.35)
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.savefig('images/Model Complexity Curve - '+ title + ' - ' + param_name + '.png')
    plt.clf() 

def plot_roc_curve(fpr, tpr, thresholds, title, prefix=None):
    
    index = title.find('Learner')
    title = title[:index]   
    plt.plot(fpr, tpr,'r-',label = 'Actual')
    #plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
    plt.plot([0,1],[0,1],'k-',label='random')
    plt.plot([0,0,1,1],[0,1,1,1],'b-',label='perfect')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    if prefix is not None:
        title = title + ' - ' + prefix
    plt.title('ROC for ' + title, fontsize = 12)
    #plt.grid(True)        
    plt.savefig('images/ROC Curve - '+ title + '.png')
    plt.clf()  

#test()

def plot_pca_old(X_pca, dataY, labels, title, prefix):
    data_ids = range(len(dataY))
    print('labels')
    print(labels)
    pass
    plt.figure(figsize=(6, 5))
    for i, c, label in zip(data_ids, 'rgbcmykw', labels):
        plt.scatter(X_pca[dataY == i, 0], X_pca[dataY == i, 1],
               c=c, label = label)

    plt.title(title)
    title = title + ' - ' + prefix
    plt.savefig('images/'+ title + '.png')
    plt.clf() 

# Referrred fr0m https://stackoverflow.com/questions/26645642/plot-multi-dimension-cluster-to-2d-plot-python
def plot_pca(X_pca, dataY, title, prefix):
    data_ids = range(len(dataY))
    print('labels')
    #print(labels)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=dataY, edgecolor='none', alpha=0.5
            )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    #plt.colorbar()
    

    plt.title(title)
    title = title + ' - ' + prefix
    plt.savefig('images/'+ title + '.png')
    plt.clf()     


def plot_kurtosis(data):
    """plt.hist(data, color=['orange'], bins='auto') 
    plt.xlabel("Value")
    plt.ylabel("Kurtosis")
    plt.grid(axis='y', alpha=0.75) 
    plt.title('Kurtosis Distribution for ICA')       
    #plt.xticks(rotation = 90)
    plt.savefig('images/kurtosis.png')
    plt.clf()  """
    ax = sns.distplot(data)
    fig = ax.get_figure()
    fig.savefig('images/kurtosis.png')



