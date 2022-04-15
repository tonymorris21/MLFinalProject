
from flask import Blueprint,render_template, redirect, url_for, request, session, Response

from models import File


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import csv
from flask import jsonify
from io import BytesIO
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use('Agg')
import base64
import collections
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sn
from flask import send_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
import missingno as msno
import time
import pyarrow.feather as feather
import json

filedata = Blueprint('filedata', __name__)

@filedata.route('/filedata/<fileid>')
def file_data(fileid):
    

    file = File.query.filter_by(fileid=fileid).first()
    session['filename'] = file.name
    session['filelocation'] = file.featherlocation
    session['fileid'] = file.fileid
    filelocation = session['filelocation']
   
    sniffer = csv.Sniffer()
    sample_bytes = 2096

   # print(hasheader)
    
    df = feather.read_feather(filelocation,use_threads=True)

    #https://matthewrocklin.com/blog/work/2017/01/12/dask-dataframes
    #https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html
    #kdnuggets.com/2020/06/machine-learning-dask.html


    plot_url = nullmatrix(file.fileid)

    contcolumns,continousdf,categoricaldf = generate_report(df)

    numcolumns = len(df.columns)

    numrows = df.shape[0]

#https://flutterq.com/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table/

    target = list(df.columns)

    df.loc[df.isnull().sum(1)>1].index

    missingfeatures = df.isnull().sum(axis=1).tolist()


    occurrences = collections.Counter(missingfeatures)

    dict(occurrences)

    columnlist = list(df.columns)
    columnlist.insert(0," ")

    catcount =  len(categoricaldf["Feature"].values)
    contcount =  len(continousdf["Feature"].values)

    catcountpt = str(catcount/(contcount+catcount)*100)
    contcountpt = str(contcount/(contcount+catcount)*100)

   # test = list(dfrows.index.values)


    
    
   


    return render_template('filedata.html',contcolumns = list(contcolumns.tolist()), plot_url=plot_url,catcount=catcount,contcount = contcount,catcountpt = catcountpt, contcountpt = contcountpt,totalrows = numrows, featurecount = numcolumns ,fileid = fileid,target = target, zip=zip,continouscolumnnames = continousdf.columns.values,continousrow_data=list(continousdf.values.tolist()),categoricalcolumnnames = categoricaldf.columns.values,categoricalrow_data=list(categoricaldf.values.tolist()))

@filedata.route('/filedata/<fileid>/nullmatrix/', methods=['GET', 'POST']) 
def nullmatrix(fileid):
    

    filelocation = session['filelocation']
    df = feather.read_feather(filelocation,use_threads=True)
    
    msno.matrix(df, sparkline=False, figsize=(11,10), fontsize=12, color=(0.27, 0.52, 1.0))
    plt.margins(0)
    img = BytesIO()
    start = time.time()
   
    
    plt.savefig(img, format='png',bbox_inches='tight')
    
    img.seek(0)
    plt.close()
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    end = time.time()
    print("The time of execution of above1 program is :", end-start)
    return plot_url

@filedata.route('/filedata/<fileid>/contvalue/<contvalue>', methods=['GET', 'POST']) 
def setContFeatures(fileid,contvalue):
 
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    data =getFeatureData(df,df[contvalue],contvalue,"Continuous")
    return data
@filedata.route('/filedata/<fileid>/catvalue/<catvalue>', methods=['GET', 'POST']) 
def setCatFeatures(fileid,catvalue):

    print("catfeatures",session.get("catfeature"))
    file_data(fileid)
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    data =getFeatureData(df,df[catvalue],catvalue,"Categorical")
    print(data)
    return data

def generate_report(df):
    
    likely_categorical = {}
    for var in df.columns:
        likely_categorical[var] = df[var].nunique()/df[var].count() < 0.05


    con = df
    continous = con.drop(list(filter(lambda x: likely_categorical[x], con.columns)), axis=1)
    cat = df.drop(continous.columns, axis=1)

    modelist = []
    for var in cat.columns:
        modelist.append(cat[var].mode()[0])

    secondmodelist = []
    for var in cat.columns:
         try:
           secondmodelist.append(cat[var].value_counts().index[1])
         except:
           secondmodelist.append(0)
             


    cat_occurrences = [cat[var].value_counts().tolist() for var in cat.columns]
    firstmodecount = [x[0] for x in cat_occurrences]
    firstmodepct = [x/cat.shape[0] for x in firstmodecount]
    firstmodepct = [x*100 for x in firstmodepct]
    secondmodecount = [x[1] if len(x) > 1 else 0 for x in cat_occurrences]
    secondmodepct = [x/cat.shape[0] for x in secondmodecount]
    secondmodepct = [x*100 for x in secondmodepct]
  
    
    
    pd.options.display.float_format = '${:,.2f}'.format
    categoricaldf = pd.DataFrame(
        {
            "Feature": array(cat.columns),
            "Count": array(cat.count()),
            "% Missing": array(cat.isna().sum()/len(cat)*100),
            "Cardinality":array(cat.nunique()),
            "Mode":modelist,
            "Mode Freq": firstmodecount,
            "Mode %":firstmodepct,
            "2nd Mode": secondmodelist,
            "2nd Mode Freq":secondmodecount,
            "2nd Mode %": secondmodepct,
            "":"Checkbox"
        }
    )
    #quartile1 = []
    #try get the std for each column in continous and store in std,catch exception and store "N/A"


    quartile1 = []
    for x in continous.columns:
        try:
            quant = continous[x].quantile(0.25)
            if quant:
                quartile1.append(quant)
        except Exception as e:
            quartile1.append("N/A")
    quartile3 = []
    for x in continous.columns:
        try:
            quant1 = continous[x].quantile(0.75)
            if quant1:
                quartile3.append(quant1)
        except Exception as e:
            quartile3.append("N/A")
    mean1 = []
    for x in continous.columns:
        try:
            mean = continous[x].mean()
            if mean:
                
                mean1.append(mean)
        except Exception as e:
            mean1.append("N/A")
    median1 = []
    for x in continous.columns:
        try:
            median = continous[x].median()
            if median:
                
                median1.append(median)
        except Exception as e:
            median1.append("N/A")
    std1 = []
    for x in continous.columns:
        try:
            std = continous[x].std()
            if std:
                std1.append(std)
        except Exception as e:
            std1.append("N/A")

    continousdf = pd.DataFrame(
        {
            "Feature": array(continous.columns),
            "Count": array(continous.count()),
            "% Missing": array(continous.isna().sum()/len(continous)*100),
            "Cardinality":array(continous.nunique()),
            "Min":array(continous.min()),
            "1st Quartile":quartile1,
            "Mean":mean1,
            "Median":median1,
            "3rd Quartile":quartile3,
            "Max":array(continous.max()),
            "Std. Deviation": std1,
            "":"Checkbox"
            
        }
    )

    continousdf = np.round(continousdf, decimals = 2)
    categoricaldf = np.round(categoricaldf, decimals = 2)

    return array(continous.columns),continousdf,categoricaldf


@filedata.route('/filedata/<fileid>/<featurename>', methods=['GET', 'POST'])

def generateGraphs(fileid,featurename):
    print("Test feature")
    datatype = request.form.get("datatype2")

    
    featurename = featurename
    filelocation = session['filelocation']
    print("filelocation",filelocation)
    df = pd.read_feather(filelocation)
    print("datatype",datatype)
    if datatype == 'continuous':
        binsize = int(np.ceil(np.log2(len(df[featurename]))) + 1)
        plot_url = generateContinuousGraphs(df[featurename],featurename,binsize)
        return plot_url
    if datatype == 'categorical':
        print("Categorical feature")
        plot_url = generateCategoricalGraphs(df[featurename],featurename)
        return plot_url




def generateContinuousGraphs(feature,featurename,binsize):
    fig = plt.figure()
    q75,q25 = np.percentile(feature, [75 ,25])
    iqr = q75 - q25
    a= sn.histplot(data=feature, stat="frequency" , bins=int(binsize),kde=True)
    a.set_title("Histogram for "+ featurename + " values")
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    p= sn.boxplot(y=feature)
    p.set_title("Box Plot for "+ featurename + " values")
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close('all')
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    
    images = plot_url + "," + plot_url2
    plt.close('all')
    return images
@filedata.route('/filedata/<fileid>/generateContinuousGraphs/feature/<feature>/binsize/<binsize>/charttype/<charttype>', methods=['GET', 'POST'])
def generateContinuousGraphsd(fileid,feature,binsize,charttype):
    fig = plt.figure()
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)

    if(charttype=="distplot"):
        a= sn.histplot(data=df[feature], stat="frequency" , bins=int(binsize),kde=True)
        a.set_title("Histogram for "+ df[feature].name)
    
        img = BytesIO()
        
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    if(charttype=="boxplot"):
        sn.boxplot(y=df[feature])
        set_title("Box Plot for "+ df[feature].name + " values")
        img = BytesIO()
    
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    if(charttype=="after changes"):
        a= sn.histplot(data=df[feature], stat="frequency" , bins=int(binsize),kde=True)
        a.set_title("Histogram for "+ df[feature].name + "after changes")
        plt.tight_layout()
        img = BytesIO()
        
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close('all')
    return plot_url

def generateCategoricalGraphs(feature,featurename):
    
    fig = plt.figure()
    
    feature.value_counts().plot(kind='bar', xlabel=featurename, ylabel='Density')
    img = BytesIO()
    plt.title("Bar plot for " + featurename)
    plt.tight_layout()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close('all')    
    return plot_url

@filedata.route('/filedata/<fileid>/correlations/<cont>', methods=['GET', 'POST'])

def correlationGraphs(fileid,cont):

    filelocation = session['filelocation']

    df = pd.read_feather(filelocation)
    plt.figure(figsize=(10, 7))
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    
    img = BytesIO()
    
    plt.suptitle("Correlation Matrix")
    
    
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    plot_url = generatescattermatrix(df,cont)
    images = plot_url + "," + plot_url2
    plt.close('all')
    return str(images)

def generatescattermatrix(df,cont):
    
    fig = plt.figure()
    cont = cont.split(',')
    cont.pop()

    df = df.sample(frac = 0.1)
    start = time.time()
    scatter_matrix(df[cont],diagonal='kde',alpha=0.5)  
    end = time.time()
    print("The time of execution of above program is :", end-start)
    img = BytesIO()
    
    plt.suptitle("Scatter Plot Matrix")
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close('all')
    return plot_url
@filedata.route('/filedata/<fileid>/dropfeature/<featurename>', methods=['GET', 'POST'])
def dropFeature(fileid,featurename):
    featurename = featurename
    print(featurename)
    print(fileid)
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    df = df.drop(featurename, 1)
    df.to_feather(filelocation)
    print("feature dropped")
    print(df)
    return redirect(url_for('filedata.file_data',fileid = fileid))

@filedata.route('/filedata/<fileid>/newgraph/<xaxis>/<yaxis>/<type>', methods=['GET', 'POST'])
def newgraph(fileid,xaxis,yaxis,type):
    type=type
    yaxis = yaxis
    xaxis = xaxis
    filelocation = session['filelocation']
    plot_url2=""
    df = pd.read_feather(filelocation)
    if type == "scatter":
        fig = plt.figure()
        sn.scatterplot(x=xaxis, y=yaxis, data=df, palette="deep")
        plt.title("Scatter Plot for "+ xaxis + " and " + yaxis)
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close('all')
    if type == "hist":
        a= sn.histplot(data=df[xaxis], stat="frequency" ,kde=True)
        plt.xlabel(xaxis)
        plt.ylabel("Frequency")
        plt.title("Histogram")
        img = BytesIO()
        plt.savefig(img,format='png')
        plt.close()
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close('all')
    if type == "boxplot":
        sn.boxplot(y=df[yaxis], x=df[xaxis], data=df)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.title("Box Plot")
        img = BytesIO()
        plt.savefig(img,format='png')
        plt.close()
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close('all')
    if type == "violin":
        sn.violinplot(x= df[xaxis],y=df[yaxis], data=df)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.title("Violin Plot")
        img = BytesIO()
        plt.savefig(img,format='png')
        plt.close()
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close('all')
    return plot_url2
@filedata.route('/filedata/<fileid>/bivariate/<target>', methods=['GET', 'POST'])
def generatebivariate(target,fileid):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    images = ""
    #for each column in dataframe generate a histplot with column as x and target as hue
    for x in df.columns:
            print(x)
            sn.histplot(x=x, hue=target, data=df,multiple="stack")
            img = BytesIO()
            plt.suptitle("Scatter plot matrix for "+x)
            plt.tight_layout()
            plt.savefig(img,format='png')
            img.seek(0)
            plt.close()
            plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
            images = images + "," + plot_url2
               
    plt.close('all')
    return str(images)

@filedata.route('/filedata/<fileid>/missingvalues/<target>', methods=['GET', 'POST'])
def getMissingFeatures(fileid,target):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    missingfeatures = df.isnull().sum(axis=1).tolist()
  
   
    occurrences = collections.Counter(missingfeatures)

    b = dict(occurrences)
    missingtarget = df[target].isnull().sum().tolist()
    #store rows in a df[column] missing values for each df[column] with column name in dictionary
    missingfeatures = df.isnull().sum(axis=0).tolist()


    missingfeatures = dict(zip(df.columns, missingfeatures))
    missingfeatures["totalrows"] = df.shape[0]

    missingtargets = str(missingtarget)
    d = dict(occurrences)
    d[99]=missingtargets
    
    return jsonify(missingfeatures)
@filedata.route('/filedata/<fileid>/dropMissingValue/<target>', methods=['GET', 'POST'])
def dropMissingFeature(fileid,target):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    df = df.dropna(how='any' ,subset=[target])
    df.reset_index(drop=True).to_feather(filelocation)
    return "success"

@filedata.route('/filedata/<fileid>/missingvalues/<featurename>/change/<change>', methods=['GET', 'POST'])
def missingValues(fileid,featurename,change):
    datatype = request.form.get("datatype")
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    print("Value changing is equal to",change)
    if change =="mean":
        df[featurename].fillna(df[featurename].mean(), inplace=True)
    if change == "mode":
        df[featurename].fillna(df[featurename].mode()[0], inplace=True)
    if change == "median":
        df[featurename].fillna(df[featurename].median(), inplace=True)
    if change == "remove":
        df.dropna(subset = [featurename], inplace=True)
    file = File.query.filter_by(fileid=fileid).first()
    df.reset_index(drop=True).to_feather(file.featherlocation)

    fig = plt.figure()
    binsize = int(np.ceil(np.log2(len(df[featurename]))) + 1)
    a= sn.histplot(data=df[featurename], stat="frequency" , bins=int(binsize),kde=True)
    a.set_title("Histogram for "+ df[featurename].name + " after changes")
    img = BytesIO()
    
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    #if datatype is eqaul to  "continuous"  then set datatype = "Continuous"
    if datatype == "continuous":
        datatype = "Continuous"

    print("datatype is",datatype)
    
    data =getFeatureData(df,df[featurename],featurename,datatype)
    dataarray = []
    dataarray.append(data)
    dataarray.append(plot_url)
    #return dataarray in json format
    return json.dumps(dataarray)
    
@filedata.route('/filedata/<fileid>/encoding/<featurename>/change/<change>', methods=['GET', 'POST'])
def encodingValues(fileid,featurename,change):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    print("Value changing is equal to",change)
    if change =="OrdinalEncoder":
        ord_enc = OrdinalEncoder()
        df[featurename] = ord_enc.fit_transform(df[[featurename]])
    if change =="LabelEncoding":
        label_enc = LabelEncoder()
        df[featurename] = label_enc.fit_transform(df[[featurename]])
    if change == "OneHotEncoder":
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(df[[featurename]]).toarray())
        enc_df.columns = enc.get_feature_names([featurename])
        df = df.join(enc_df)

    file = File.query.filter_by(fileid=fileid).first()
    df.to_feather(file.featherlocation)
    print(df.head)
    data =getFeatureData(df,df[featurename],featurename,"Categorical")
    return data
@filedata.route('/filedata/<fileid>/downloadcleandata')
def downloaddata(fileid):
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_feather(file.featherlocation)
    #convert df to csv
    csv = df.to_csv(index=False)

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=data.csv"})

@filedata.route('/filedata/<fileid>/clamp/<featurename>/datatype/<datatype>/upper/<uppervalue>/lower/<lowervalue>', methods=['GET', 'POST'])
def clampTransformation(fileid, featurename,datatype,uppervalue,lowervalue):
    file = File.query.filter_by(fileid=fileid).first()
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    print("Lower value",lowervalue)
    df[featurename] = df[featurename].clip(upper=float(uppervalue),lower=float(lowervalue))
    df.to_feather(file.featherlocation)
    data =getFeatureData(df,df[featurename],featurename,"Continuous")
    return data
def getFeatureData(df,feature,featurename,datatype):
    if datatype=="Continuous":
        data = []
        float_formatter = "{:.2f}".format
        data.append(featurename)
        data.append(int(feature.count()))
        data.append(feature.isna().sum()/len(feature)*100)#missing
        data.append(int(feature.nunique()))#cardinality
        data.append(float_formatter(float(feature.min())))#min
        data.append(float_formatter(feature.quantile(0.25)))#1st
        data.append(float_formatter(feature.mean()))#mean
        data.append(float_formatter(feature.median()))#median
        data.append(float_formatter(feature.quantile(0.75)))#3rd
        data.append(float_formatter(feature.max()))#max
        print("max",feature.max())
        data.append(float_formatter(feature.std()))#std
        print(data)
        return json.dumps(data)
    else:
        data = []
        float_formatter = "{:.2f}".format
        
        
        d =df[featurename].value_counts().index.tolist()
        f = df[featurename].value_counts().tolist()
        data.append(featurename)
        data.append(int(feature.count()))
        data.append(feature.isna().sum()/len(feature)*100)
        data.append(int(feature.nunique()))
        #check if f[i] is equal to string
        for i in range(len(f)):
            if type(f[i]) is str:
                f[i] = float(f[i])
        data.append(d[0])
        data.append(float_formatter(f[0]))
        
        data.append(float_formatter(f[0]/len(feature)*100))
        data.append(d[1])
        data.append(float_formatter(f[1]))
        data.append(float_formatter(f[1]/len(feature)*100))
        print(data)
        return json.dumps(data)

@filedata.route('/filedata/whiskerValues/Q1/<Q1>/Q3/<Q3>', methods=['GET', 'POST'])
def whiskerValues(Q1,Q3):
    Q1 = Q1.strip()
    Q3 = Q3.strip()
    uppervalue = float(Q3)+(1.5*(float(Q3)-float(Q1)))
    lowervalue=float(Q1)-1.5*(float(Q3)-float(Q1))
    values = str(uppervalue) + "," + str(lowervalue)
    return values
