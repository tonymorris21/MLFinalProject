import os
from flask import Blueprint,render_template, request, session

from __init__ import db
from models import File,Model

from flask import current_app
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
import base64
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import uuid
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing,metrics

from datetime import date
from urllib import parse

from flask import send_file
from bs4 import BeautifulSoup
import seaborn as sn
from sklearn import preprocessing
import json

train = Blueprint('train', __name__)


@train.route('/train/<fileid>/target/<target>', methods=['POST','GET'])
def trainpage(fileid,target):
    return render_template('algorithmselect.html')

@train.route('/train/<fileid>/algorithm/<algorithm>/target/<target>/param/<param>', methods=['POST','GET'])
def train_data(fileid,algorithm,target,param):
    
    parsed = parse.unquote(param)
    parsed= parsed.replace("{","")
    parsed= parsed.replace("}","")
    parsed= parsed.replace('"','')
    parsed = parsed.split(",")

    elem = dict()
    for parsed in parsed:
        elem1 = parsed.split(":")
        elem[elem1[0]] = elem1[1]

    file = File.query.filter_by(fileid=fileid).first()
    dfd = pd.read_csv(file.location)
    df = pd.read_feather(file.featherlocation)
    print("test",)

    X = df.drop(target, 1)
    Y = df.filter([target])
    classnames = dfd[target].unique()
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize
    print("test",classnames)
    #if key 'Normalisation' is present in elem, then use it as Normalisation
    featurenames = list(X.columns.values)
    if 'Normalisation' in elem:
        if elem['Normalisation'] == 'yes':
            X = preprocessing.normalize(X)


    if 'randomstate' in elem:
        randomstate = elem['randomstate']
    else:
        randomstate = 1
    #if key 'split' is present in elem, then use it as split
    print(elem.keys())
    if 'split' in elem.keys():
        split = float(elem['split'])/100
        print("Split is",split)
    else:
        split = 0.25
    print("Split is",split)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=split, random_state=int(randomstate))

    # if key 'Kernel' is present in elem, then use it as kernel
    if 'Kernel' in elem:
        Kernel = elem['Kernel']
    else:
        Kernel = 'rbf'
    # if key 'C' is present in elem, then use it as C
    if 'k' in elem:
        k = elem['k']
    else:
        k = 5

    # if key 'var_smoothing' is present in elem, then use it as var_smoothing
    if 'var_smoothing' in elem:
        var_smoothing = elem['var_smoothing']
    else:
        var_smoothing = 1e-09


    #if key 'ntree' is present in elem, then use it as ntree
    if 'ntree' in elem:
        ntree = elem['ntree']
    else:
        ntree = 100
    #if key 'max_features' is present in elem, then use it as max_features
    if 'max_features' in elem:
        max_features = elem['max_features']
    else:
        max_features = 'auto'


    if algorithm == "RF" : modelid, plot_url, accuracy, tables = RFC(max_features,ntree,
    dfd[target],X_train,X_test,y_train,y_test,fileid,target,featurenames)
    elif algorithm == "NB" : modelid, plot_url, accuracy, tables = naivebayes(var_smoothing,
    dfd[target],X_train,X_test,y_train,y_test,fileid,target,featurenames)
    elif algorithm == "KNN" : modelid, plot_url, accuracy, tables = KNN(k,
    dfd[target],X_train,X_test,y_train,y_test,fileid,target,featurenames)
    elif algorithm == "SVC" : modelid, plot_url, accuracy, tables = svc(Kernel,
    dfd[target],X_train,X_test,y_train,y_test,fileid,target,featurenames)

    return render_template('modelinfo.html', modelid=modelid,confusion_matrix=plot_url,accuracy=accuracy,tables=tables,algorithm=algorithm,target=target)



def naivebayes(var_smoothing,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):
    gaussian = GaussianNB(var_smoothing=var_smoothing)
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    accuracy = accuracy_score(y_test,Y_pred)
    algorithm = "NB"
    classnames = targetdf.unique()
    modelid, plot_url, accuracy, tables = toDatabase(feature_names,
    target,algorithm,classnames,gaussian,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def KNN(k,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_feather(file.featherlocation)

    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    knn.feature_names = feature_names
    Y_pred = knn.predict(X_test) 

    accuracy = accuracy_score(y_test,Y_pred)


    algorithm = "KNN"
    modelid, plot_url, accuracy, tables = toDatabase(feature_names,target,algorithm,targetdf.unique(),knn,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables
def RFC (max_features,ntree,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):


    rf = RandomForestClassifier(max_features=max_features,n_estimators=ntree)
    rf.fit(X_train, y_train)
    rf.score(X_test, y_test)
    rf.score(X_train,y_train)
    Y_pred = rf.predict(X_test) 

    accuracy = accuracy_score(y_test,Y_pred)

    algorithm = "RF"
    modelid, plot_url, accuracy, tables = toDatabase(feature_names,target,algorithm,targetdf.unique(),rf,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def svc(Kernel,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):

    svc = SVC(probability=False,kernel=Kernel)
    svc.fit(X_train, y_train)
    svc.feature_names = feature_names
    svc.score(X_test, y_test)
    svc.score(X_train,y_train)
    Y_pred = svc.predict(X_test) 
    accuracy = accuracy_score(y_test,Y_pred)

    cm = confusion_matrix(y_test,Y_pred)
    sn.heatmap(cm,annot=True,cmap="Greens")

    plt.show()
    algorithm = "SVC"
    modelid, plot_url, accuracy, tables = toDatabase(feature_names,target,algorithm,targetdf.unique(),svc,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def toDatabase(feature_names,target,algorithm,classnames,model,accuracy,y_test,X_test,Y_pred):

    cv = classification_report(y_test, Y_pred)
    classreport = report_to_df(cv)
    cm = confusion_matrix(y_test, Y_pred)
    ax= plt.subplot()
    sn.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classnames)
    ax.yaxis.set_ticklabels(np.flip(classnames))
    img = BytesIO()
    plt.title('Confusion matrix for our classifier')
    plt.savefig(img, format='png')
    plt.close()


    img.seek(0)


    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    projectid = session['projectid']
    createddate = date.today()
    tables=[classreport.to_html(index=False,classes='table table-sm')]
    class_report = str(tables)
    projectid = session['projectid']
    filename = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models\\', str(uuid.uuid4())+'_model.sav')
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    classnames = list(classnames)
    classnames = json.dumps(classnames)
    feature_names = json.dumps(feature_names)
    model = Model(projectid=projectid, location = filename,
    feature_names=feature_names,target=target,algorithm=algorithm,classnames=classnames,
    createddate=createddate,accuracy=accuracy,confusion_matrix=plot_url,class_report=class_report)
    db.session.add(model)
    db.session.commit()
    modelid = model.modelid


    return modelid,plot_url, accuracy, tables

def classnamemap(le,inverse):
    classnames = dict(zip(le, inverse))
    return classnames

@train.route('/modelEvaluation/<modelid>', methods=['POST','GET'])
def modelinfo(modelid):
    model = Model.query.filter_by(modelid=modelid).first()
    confusion_matrix = model.confusion_matrix
    accuracy = model.accuracy
    classreport = model.class_report
    classreport = str(classreport).replace('[', '')
    classreport = str(classreport).replace(']', '')
    classreport = str(classreport).replace("'", '')
    classreport = str(classreport).replace("\\n", '')
    classreport.strip('/n')
    soup = BeautifulSoup(classreport)
    soup.get_text(strip=True)
    return render_template('modelinfo.html',modelid=model.modelid,target = model.target,algorithm = model.algorithm,accuracy=accuracy,confusion_matrix=confusion_matrix,tables=soup)



def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df

@train.route('/train/<modelid>/downloadModel', methods=['POST','GET'])
def downloadModel(modelid):
    model = Model.query.filter_by(modelid=modelid).first()
    
    return send_file(model.location,
                     mimetype='application/octet-stream',
                     attachment_filename='data.pickle',
                     as_attachment=True)