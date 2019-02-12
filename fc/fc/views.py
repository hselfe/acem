from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import io
import urllib, base64
import seaborn as sns

def home(request):
    return render(request, 'home.html')

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def display_cleanly(item_in):
    str_out = ''
    for a, b in item_in.items(): # iterating freqa dictionary
        str_out = str_out + str(a) + ' : ' + str(b) + '<br />'
    return str_out

def churn(request):
    url = "https://cxmodel4909235007.blob.core.windows.net/futurecarsales/churn.csv?sp=r&st=2019-02-12T11:07:50Z&se=2019-02-12T19:07:50Z&spr=https&sv=2018-03-28&sig=FjC3hY56mDJpeFXheTF3dnvQWniYAaEjf8GuE1tHCZM%3D&sr=b"
    data = pd.read_csv(url)
    data.head()
    data['Gender'].replace(['Male','Female'],[0,1],inplace=True)
    data['Partner'].replace(['Yes','No'],[1,0],inplace=True)
    data['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan1'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan2'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan3'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan4'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan5'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan6'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan7'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan8'].replace(['Yes','No'],[1,0],inplace=True)
    data['ServicePlan9'].replace(['Yes','No'],[1,0],inplace=True)
    data['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)
    data['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
    data['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
    data['Churn'].replace(['Yes','No'],[1,0],inplace=True)
    data.pop('customerID')
    data.pop('TotalCharges')
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
    heat_map=plt.gcf()
    heat_map.set_size_inches(20,15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Correlation between service plans and churn')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    plt.clf()

    top_4 = get_top_abs_correlations(data, 4)
    top_4 = display_cleanly(top_4)
    return render(request, 'churn.html', {'uri': uri, 'top_4': top_4, 'url': url})

def count(request):
    url = "https://cxmodel4909235007.blob.core.windows.net/futurecarsales/Social_Network_Ads.csv?sp=r&st=2019-02-12T10:59:06Z&se=2019-02-12T18:59:06Z&spr=https&sv=2018-03-28&sig=SiH0QB2gVeGbAi6lnPeMh7DCMvzUT8DWncDY9NGPCtY%3D&sr=b"
    dataset = pd.read_csv(url)
    data_shape = dataset.shape
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    c=0
    for i in range(0,len(y_pred)):
        if(y_pred[i]==y_test[i]):
            c=c+1
    accuracy=c/len(y_pred)
    cm = confusion_matrix(y_test, y_pred)

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Nearest Neighbours - Accuracy')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    plt.clf()

    return render(request, 'count.html', {'data_shape': data_shape, 'accuracy': accuracy*100,'cm': cm, 'uri2': uri, 'url': url})
