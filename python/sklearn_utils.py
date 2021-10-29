#!/usr/bin/env python3
#############################################################################
### sklearn_utils.py - Scikit-Learn utilities
### 
### http://scikit-learn.org/
### 
### Dataset: (X,y)
###   X = float array, N samples * n features
###   y = integer labels, N * (1 or 0)
#############################################################################
import sys,os,time,re,argparse,logging,random
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as mpl_pyplot
import matplotlib.colors as mpl_colors #ListedColormap

import sklearn.metrics
import sklearn.model_selection
from sklearn.datasets import make_classification as skl_make_classification
from sklearn.feature_extraction import DictVectorizer as skl_DictVectorizer
from sklearn.preprocessing import StandardScaler as skl_StandardScaler
from sklearn.ensemble import RandomForestClassifier as skl_RandomForestClassifier, AdaBoostClassifier as skl_AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as skl_KNeighborsClassifier
from sklearn.svm import SVC as skl_SVC
from sklearn.tree import DecisionTreeClassifier as skl_DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB as skl_GaussianNB
from sklearn.neural_network import BernoulliRBM as skl_BernoulliRBM, MLPClassifier as skl_MLPClassifier
from sklearn.decomposition import PCA as skl_PCA

import csv #replace with pandas
#csv.register_dialect("skl", strict=True, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

##############################################################################
#def CsvDialect(name):
# return csv.get_dialect(name)
#
##############################################################################
def ClassifierFactory(args):
  clf=None;
  alg = args.alg.upper();
  if alg=='AB':
    clf = skl_AdaBoostClassifier(algorithm='SAMME.R')
  elif alg=='DT':
    clf = skl_DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None)
  elif alg=='KNN':
    clf = skl_KNeighborsClassifier(n_neighbors=4, algorithm='auto', metric='minkowski', p=2)
  elif alg=='MLP':
    clf = skl_MLPClassifier(hidden_layer_sizes=(args.nn_layers, ), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=args.nn_max_iter, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  elif alg=='NB':
    clf = skl_GaussianNB()
  elif alg=='RF':
    clf = skl_RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
  elif alg=='SVM':
    try:
      gamma = float(args.svm_gamma)
    except:
      logging.debug(f"args.svm_gamma = '{args.svm_gamma}'")
      gamma = 'auto'
    clf = sklearn_utils.SVMClassifierFactory(kernel=args.svm_kernel, cparam=args.svm_cparam, gamma=gamma)
  return clf

##############################################################################
def SVMClassifierFactory(kernel, cparam=1.0, gamma='auto'):
  clf = skl_SVC(kernel=kernel, C=cparam, gamma=gamma)
  return clf

##############################################################################
#def ReadDataset(fin, eptag=None, ignore_tags=None, csvdialect=CsvDialect('skl')):
def ReadDataset(fin, delim, eptag=None, ignore_tags=None):
  '''Read from file.  Classification or regression.  All features and endpoint must be numeric.'''
  n_data=0; n_col=0; X=[]; y=[];
  #csvReader=csv.DictReader(fin, dialect=csvdialect, fieldnames=None)
  #logging.debug('n_fieldnames: %d'%len(csvReader.fieldnames))
  #logging.debug('fieldnames = %s'%str(csvReader.fieldnames))
  df = pd.read_csv(fin, sep=delim)

  if not eptag:
    #eptag = csvReader.fieldnames[-1]
    eptag = df.columns[-1]
  #j_eptag = csvReader.fieldnames.index(eptag)
  j_eptag = df.columns.index(eptag)

  ignore_tags = re.split(r'\s*,\s*', ignore_tags.strip()) if ignore_tags else []
  #featuretags = csvReader.fieldnames[:]
  featuretags = df.columns[:]
  featuretags.pop(j_eptag)
  for tag in ignore_tags:
    featuretags.remove(tag)
  logging.debug(f"eptag = '{eptag}'; j_eptag = {j_eptag}")
  logging.debug(f"featuretags = {str(featuretags)}")

  #while True:
  #  try:
  #    row = next(csvReader)
  #    row_featurevals = [row[tag] for tag in featuretags]
  #    row_epval = row[eptag]
  #    X.append(row_featurevals)
  #    y.append(row_epval)
  #    n_data+=1
  #  except Exception as e:
  #    break #normal EOF

  #X = np.array(X)
  #y = np.array(y)

  for tag in featuretags:
    try:
      df[tag] = df[tag].astype(float, errors="raise")
    except Exception as e:
      logging.error(f"df['{tag}'].astype(float) failed: {e}")

  try:
    df[eptag] = df[eptag].astype(int, errors="raise")
  except Exception as e:
    logging.error(f"df['{eptag}'].astype(int) failed: {e}")

  X = np.array(df[featuretags])
  y = np.array(df[eptag])

  logging.debug(f"n_X = {X.shape[0]}")
  logging.debug(f"n_y = {len(y)}")
  logging.debug(f"X.shape = {X.shape[1]}")

  if X.shape[0]!=len(y):
    logging.error(f"n_X != n_y; {X.shape[0]} != {len(y)}")

  return (X,y,featuretags,eptag)

##############################################################################
#def VectorizeCSV(fin, fout, coltags, csvdialect):
# '''What to do with missing values?  quoting=csv.QUOTE_NONNUMERIC means all non-quoted fields converted to float.'''
def VectorizeCSV(fin, fout, coltags):
  n_in=0;  n_out=0;
  #csvReader = csv.DictReader(fin, dialect=csvdialect, fieldnames=None)
  df = pd.read_csv(fin, sep=delim)
  #logging.debug('fieldnames: %d'%len(csvReader.fieldnames))
  #logging.debug('fieldnames = %s'%str(csvReader.fieldnames))
  logging.debug(f"fieldnames: ({df.shape[1]}) {df.columns}")

  vec = skl_DictVectorizer()

  #rows=[];
  #while True:
  #  try:
  #    row = next(csvReader)
  #    n_in+=1
  #    rows.append(row)
  #  except csv.Error as e:
  #    logging.debug('bad row: %s'%e)
  #    continue
  #  except Exception as e:
  #    if 'could not convert string to float' in str(e):
  #      logging.debug('bad row: %s'%e)
  #      continue
  #    else:
  #      break #normal EOF
  #rows_vectorized = vec.fit_transform(rows)

  rows_vectorized = vec.fit_transform(df.to_dict())

  logging.debug(f"vectorized fieldnames: {len(vec.get_feature_names())}")
  logging.debug(f"vectorized fieldnames = {str(vec.get_feature_names())}")

  #scipy.sparse.csr.csr_matrix
  #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
  logging.debug(f"type(rows_vectorized): {type(rows_vectorized)}")
  logging.debug(f"rows read: {n_in}")

  rv = rows_vectorized.todok()

  #csvWriter=csv.DictWriter(fout, fieldnames=vec.get_feature_names(), dialect=csvdialect)
  #csvWriter.writeheader()
  #tags = vec.get_feature_names()
  #for i in range(rv.get_shape()[0]):
  #  row_this={}
  #  for j,tag in enumerate(tags):
  #    key = (i,j)
  #    row_this[tag] = rv[key] if key in rv else 0
  #  csvWriter.writerow(row_this)
  #  n_out+=1
  #logging.debug('CSV rows out: %d'%(n_out))

  pd.DataFrame(rv).to_csv(fout, "\t", index=False)

##############################################################################
#def CheckCSV(fin, csvdialect):
def CheckCSV(fin, csvdialect):
  '''Ok for dataset?  All numeric?  Or needing vectorization.'''
  try:
    #X,y,ftags,etag = ReadDataset(fin, csvdialect=csvdialect)
    X,y,ftags,etag = ReadDataset(fin, delim, eptag=None, ignore_tags=None)
  except Exception as e:
    logging.error(e)
  if X.shape[0]>0 and X.shape[1]>0:
    logging.info(f"Dataset ok, with N_cases = {X.shape[0]} and N_features = {X.shape[1]}")
    return True
  elif X.shape[0]==0:
    logging.error('Dataset not ok, with N_cases = 0')
    return False
  elif X.shape[1]==0:
    logging.error('Dataset not ok, with N_features = 0')
    return False
  else:
    logging.error('Dataset not ok.')
    return False

##############################################################################
#def SplitCSV(fin, fout, fout_split, split_pct, csvdialect=CsvDialect('skl')):
def SplitCSV(fin, fout, delim, fout_split, split_pct):
  #n_in=0; n_train=0; n_test=0;
  #csvReader = csv.DictReader(fin, dialect=csvdialect, fieldnames=None)
  #logging.debug('fieldnames: %d'%len(csvReader.fieldnames))
  #logging.debug('fieldnames = %s'%str(csvReader.fieldnames))
  #csvWriter = csv.DictWriter(fout, fieldnames=csvReader.fieldnames, dialect=csvdialect)
  #csvWriter_split = csv.DictWriter(fout_split, fieldnames=csvReader.fieldnames, dialect=csvdialect)
  #csvWriter.writeheader()
  #csvWriter_split.writeheader()
  #while True:
  #  try:
  #    row = next(csvReader)
  #    n_in+=1
  #  except csv.Error as e:
  #    logging.debug('bad row: %s'%e)
  #    continue
  #  except Exception as e:
  #    if 'could not convert string to float' in str(e):
  #      logging.debug('bad row: %s'%e)
  #      continue
  #    else:
  #      break #normal EOF
  #  if 100.0*random.random() < split_pct:
  #    csvWriter_split.writerow(row)
  #    n_test+=1
  #  else:
  #    csvWriter.writerow(row)
  #    n_train+=1
  #n_out=n_train+n_test
  #logging.debug('CSV rows read: %d'%(n_in))
  #logging.debug('CSV rows out (train): %d (%.1f%%)'%(n_train, 100.0*n_train/n_out))
  #logging.debug('CSV rows out (test): %d (%.1f%%)'%(n_test, 100.0*n_test/n_out))
  #logging.debug('CSV rows out (TOTAL): %d'%(n_out))

  df = pd.read_csv(fin, sep=delim)
  logging.debug(f"fieldnames: ({df.shape[1]}) {df.columns}")

  df_train = df.sample(frac=split_pct/100, replace=False)
  df_test = df[list(set(df.index) - set(df_train.index))]
  logging.info(f"rows read: {df.shape[0]}")
  logging.info(f"rows out (train): {df_train.shape[0]} ({100.0*df_train.shape[0]/df.shape[0]:%.1f}%)") 
  logging.info(f"rows out (test): {df_test.shape[0]} ({100.0*df_test.shape[0]/df.shape[0]:.1f}%)")
  logging.info(f"rows out (TOTAL): {df.shape[0]}")

  #return n_in,n_train,n_test
  return df.shape[0],df_train.shape[0],df_test.shape[0]

##############################################################################
def SplitDataset(X, y, split_pct):
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=split_pct/100)
  logging.debug('split: total: %d ; train : %d (%.1f%%) ; test: %d (%.1f%%)'%(X.shape[0],
	X_train.shape[0], 100.0*X_train.shape[0]/X.shape[0],
	X_test.shape[0], 100.0*X_test.shape[0]/X.shape[0]))
  return X_train, X_test, y_train, y_test

##############################################################################
def GenerateRandomDataset(nclass, nsamp, nfeat, fout):
  X,y = skl_make_classification(
        n_classes=nclass,
        n_samples=nsamp,
        n_features=nfeat,
        n_redundant=0,
        n_informative=2,
        random_state=random.randint(0, 100),
        n_clusters_per_class=1)

  fout.write(('\t'.join([f"f{j}" for j in range(1, nfeat+1)]))+"\tlabel\n")
  for i,row in enumerate(X):
    fout.write(('\t'.join(map(lambda x:f"{x:.2f}", row)))+(f"\t{y[i]}\n"))

##############################################################################
def TestClassifier(clf, X, y, name, fout):
  MESSAGES=[];
  MESSAGES.append(f"{name:12s}: Nsamp: {len(X):6d} ; Nfeat: {X.shape[1]:6d} ; Nclas: {len(set(y)):6d}")

  if len(set(y))>2:
    logging.error('Only handles Nclasses = 2.')
    return

  mean_acc = clf.score(X, y) #score() returns mean accuracy.

  y_pred = clf.predict(X)

  cmat = sklearn.metrics.confusion_matrix(y, y_pred)
  #logging.debug('cmat=%s'%(str(cmat)))
  #logging.debug('cmat.ravel()=%s'%(str(cmat.ravel())))
  tn, fp, fn, tp = cmat.ravel()
  MESSAGES.append(f"{name:<12}: tp = {tp:6d} ; fp = {fp:6d} ; tn = {tn:6d} ; fn = {fn:6d}")
  prec = sklearn.metrics.precision_score(y, y_pred)
  rec = sklearn.metrics.recall_score(y, y_pred)
  MESSAGES.append(f"{name:<12}: mean_accuracy = {mean_acc:.4f} ; precision = {prec:.4f} ; recall = {rec:.4f}")
  mcc = sklearn.metrics.matthews_corrcoef(y, y_pred)
  f1 = sklearn.metrics.f1_score(y, y_pred)
  MESSAGES.append(f"{name:<12}: F1_score = {f1:.4f} ; MCC = {mcc:.4f}")

  if fout:
    fout.write(("\t".join(["f{j}" for j in range(1, X.shape[1]+1)]))+"\tlabel\n")
    for i in range(len(X)):
      fout.write(("\t".join(map(lambda x:"{x:.3f}", X[i])))+("\t{y_pred[i]}\n"))
  else:
    MESSAGES.append(f"{name:<12}: No output file.")

  for line in MESSAGES:
    logging.debug(line)

  return MESSAGES

##############################################################################
def PlotPCA(clf, X_train, y_train, X_test, y_test, cnames, epname, title, subtitle, width, height, dpi, ofile):
  '''Projecting the feature space for all cases (training and test) onto 2D via PCA.'''

  # True vs. false predictions:
  y_test_predicted = clf.predict(X_test)
  y_test_t = (y_test_predicted == y_test) #boolean array
  y_test_f = np.logical_not(y_test_t) #boolean array
  n_test_t = np.where(y_test_t)[0].shape[0]
  n_test_f = np.where(y_test_f)[0].shape[0]

  y_test_tp =  (y_test_t & np.array(y_test_predicted, dtype=bool)) #boolean array
  y_test_tn =  (y_test_t & np.logical_not(y_test_predicted)) #boolean array
  n_test_tp = np.where(y_test_tp)[0].shape[0]
  n_test_tn = np.where(y_test_tn)[0].shape[0]

  y_test_fp =  (y_test_f & np.logical_not(y_test_predicted)) #boolean array
  y_test_fn =  (y_test_f & np.array(y_test_predicted, dtype=bool)) #boolean array
  n_test_fp = np.where(y_test_fp)[0].shape[0]
  n_test_fn = np.where(y_test_fn)[0].shape[0]

  logging.info('Test-predictions: T = %3d ; TP = %3d ; TN = %3d'%(n_test_t, n_test_tp, n_test_tn))
  logging.info('Test-predictions: F = %3d ; FP = %3d ; FN = %3d'%(n_test_f, n_test_fp, n_test_fn))

  pca_d = 2
  pca = skl_PCA(n_components=pca_d)

  X = np.concatenate((X_train, X_test), axis=0)
  y = np.concatenate((y_train, y_test), axis=0)

  n_train = X_train.shape[0]
  n_test = X_test.shape[0]

  pca.fit(X)
  X_r = pca.transform(X)

  logging.debug('X.shape = %s ; X_r.shape = %s'%(str(X.shape), str(X_r.shape)))
  logging.info('PCA %dD to %d-component'%(X.shape[1], X_r.shape[1]))
  logging.info('PCA explained variance ratio (1st 2 components): %s'%str(pca.explained_variance_ratio_))

  fig = mpl_pyplot.figure(frameon=False, tight_layout=False)
  fig.set_size_inches((width, height))
  fig.set_dpi(dpi)
  ax = fig.gca()

  logging.debug('figure size: %s ; DPI: %d'%(str(fig.get_size_inches()), fig.dpi))

  colors = ['navy', 'turquoise', 'darkorange', 'forestgreen']
  lw = 1

  y_vals = list(set(y))
  y_vals.sort()

  if cnames:
    ylabels = [cnames[int(y_val)] for y_val in y_vals]
  else:
    ylabels = [str(int(y_val)) for y_val in y_vals]

  X_r_train = X_r[:n_train]
  X_r_test = X_r[-n_test:]
  for y_val, color, ylabel in zip(y_vals, colors, ylabels):
    mpl_pyplot.scatter(X_r_train[y_train==y_val, 0], X_r_train[y_train==y_val, 1], color=color, marker='.', alpha=.8, lw=lw, label='train:'+ylabel) #Train
    #mpl_pyplot.scatter(X_r_test[y_test==y_val, 0], X_r_test[y_test==y_val, 1], color=color, marker='+', alpha=.8, lw=lw, label='test:'+ylabel) #Test

  X_r_test_fp = X_r_test[np.where(y_test_fp)]
  mpl_pyplot.scatter(X_r_test_fp[:, 0], X_r_test_fp[:, 1], color='red', marker='^', alpha=.5, lw=lw, label='test:FP') #red-^ FPs
  X_r_test_fn = X_r_test[np.where(y_test_fn)]
  mpl_pyplot.scatter(X_r_test_fn[:, 0], X_r_test_fn[:, 1], color='red', marker='v', alpha=.5, lw=lw, label='test:FN') #red-v FNs

  X_r_test_tp = X_r_test[np.where(y_test_tp)]
  mpl_pyplot.scatter(X_r_test_tp[:, 0], X_r_test_tp[:, 1], color='green', marker='^', alpha=.5, lw=lw, label='test:TP') #green-^ TPs
  X_r_test_tn = X_r_test[np.where(y_test_tn)]
  mpl_pyplot.scatter(X_r_test_tn[:, 0], X_r_test_tn[:, 1], color='green', marker='v', alpha=.5, lw=lw, label='test:TN') #green-v TNs

  mpl_pyplot.legend(loc='best', title=None, shadow=True, scatterpoints=1)
  mpl_pyplot.title('PCA (%dD) of %s\n%s'%(pca_d, title, (subtitle if subtitle else '')))
  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  ax.set_xticklabels(ax.get_xticklabels(), size='small') #ticklabels go away?
  ax.set_yticklabels(ax.get_yticklabels(), size='small') #ticklabels go away?
  ax.annotate('%s'%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
        xycoords='axes fraction', xy=(0.0, -0.1),
        horizontalalignment='left', verticalalignment='top')
  ax.annotate('N_cases = %d\nN_features = %d'%(X.shape[0], X.shape[1]),
        xycoords='axes fraction', xy=(0.8, 0.5),
        horizontalalignment='center', verticalalignment='top')
  ax.annotate('N_train = %d (%.1f%%)\nN_test = %d (%.1f%%)'%(n_train, n_train*100/X.shape[0], n_test, n_test*100/X.shape[0]),
        xycoords='axes fraction', xy=(0.8, 0.35),
        horizontalalignment='center', verticalalignment='top')
  ax.annotate('TP = %d, TN = %d\nFP = %d, FN = %d'%(n_test_tp, n_test_tn, n_test_fp, n_test_fn),
        xycoords='axes fraction', xy=(0.8, 0.2),
        horizontalalignment='center', verticalalignment='top')
  ax.annotate('mean_accuracy = %.1f%%'%(clf.score(X_test, y_test)*100),
        xycoords='axes fraction', xy=(0.8, 0.05),
        horizontalalignment='center', verticalalignment='top')

  ### Decision boundary, mapped to 2D PCA:
  mesh_h = .2  # mesh step size
  xplot_min = X_r[:, 0].min()
  xplot_max = X_r[:, 0].max()
  yplot_min = X_r[:, 1].min()
  yplot_max = X_r[:, 1].max()

  #xplot_mesh, yplot_mesh = np.meshgrid(np.arange(xplot_min, xplot_max, mesh_h), np.arange(yplot_min, yplot_max, mesh_h))
  xplot_mesh = X_r[:, 0]
  yplot_mesh = X_r[:, 1]

  #Need more colors for n_classes > 2 ?
  cm_contour = mpl_pyplot.cm.RdBu
  cm_bright = mpl_colors.ListedColormap(['#FF0000', '#0000FF'])

  #Assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
  #Decision function returns how far from decision surface, sign indicating which side.
  if hasattr(clf, "decision_function"):
    #ax2.set_title('%s: decision_function contours'%(title))
    #X_r = pca.transform(X)
    Z = clf.decision_function(X)
  else:
    #ax2.set_title('%s: prediction contours'%(title))
    Z = clf.predict_proba(X)

  # Put the result into a color plot
  Z = Z.reshape(xplot_mesh.shape)

  ### FIX THIS...
  #logging.debug('Z.shape = %s'%(str(Z.shape)))
  #ax.contourf(xplot_mesh, yplot_mesh, Z, cmap=cm_contour, alpha=.8)

  ###
  if ofile:
    fig.savefig(ofile)

  return fig

##############################################################################
def CrossValidate(clf,X,y,cv_folds):
  cv_scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=cv_folds, verbose=bool(logging.getLogger().getEffectiveLevel()<=logging.DEBUG))
  logging.info(f"cv_scores={str(cv_scores)}")

##############################################################################
def Demo(clf,nclass,nfeat,nsamp,show_plot,ofile_plot):
  '''Demo classifier using randomly generated dataset for training and test.'''
  ### Example dataset: two interleaving half circles
  #X, y = sklearn.datasets.make_moons(noise=0.3, random_state=0)
  ### Example dataset: 
  #X, y = sklearn.datasets.load_diabetes() #regression
  #X, y = sklearn.datasets.load_breast_cancer() #classification

  nclass = nclass if nclass else 2
  nfeat = nfeat if nfeat else 2 #allows 2D plot
  nsamp = nsamp if nsamp else random.randint(50,200)

  ###Generate random classification dataset
  X, y = skl_make_classification(
        n_classes=nclass,
        n_samples=nsamp,
        n_features=nfeat,
        n_redundant=0,
        n_informative=2,
        random_state=random.randint(0,100),
        n_clusters_per_class=1)

  # Preprocess dataset, split into training and test part
  X = skl_StandardScaler().fit_transform(X)
  #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.4)
  X_train, X_test, y_train, y_test = sklearn_utils.SplitDataset(X, y, split_pct=25)

  ### Train the classifier:
  clf.fit(X,y)

  ### Test the classifier:
  sklearn_utils.TestClassifier(clf,X_train,y_train,'train',None)
  sklearn_utils.TestClassifier(clf,X_test,y_test,'test',None)
  sklearn_utils.TestClassifier(clf,X,y,'train_and_test',None)

  fnames = ['feature_%02d'%(j+1) for j in range(nfeat)]
  epname = 'endpoint'

  title = re.sub(r"^.*'.*\.(.*)'.*$",r'\1',str(type(clf)))

  if show_plot or ofile_plot:
    if nfeat>2:
      sklearn_utils.PlotPCA(clf,X_train,y_train,X_test,y_test,fnames,epname,title,None,7,5,100,ofile_plot)
    else:
      PlotClassifier(clf,X_train,y_train,X_test,y_test,fnames,epname,title,None,ofile_plot)
    if show_plot:
      mpl_pyplot.show()

##############################################################################
def PlotClassifier(clf,X_train,y_train,X_test,y_test,fnames,epname,title,subtitle,ofile):
  '''Only for n_features = 2,  n_classes = 2.'''
  logging.debug("PLOT: "+title)
  mesh_h = .02  # mesh step size
  figsize = (12,8) #w,h in inches
  fig = mpl_pyplot.figure(figsize=figsize, dpi=100, frameon=False, tight_layout=False)

  X = np.concatenate((X_train,X_test),axis=0)
  x_min,x_max = X[:,0].min()-.5, X[:,0].max()+.5
  y_min,y_max = X[:,1].min()-.5, X[:,1].max()+.5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_h),
                          np.arange(y_min, y_max, mesh_h))
  my_colors = ['#FF0000', '#0000FF', '#00FF00', '#999999']

  #Need more colors for n_classes > 2 ?
  cm_contour = mpl_pyplot.cm.RdBu
  cm_bright = mpl_colors.ListedColormap(my_colors)

  ### Axes 1: dataset points only.
  ax1 = mpl_pyplot.subplot(1, 2, 1)
  ax1.set_title('%s: dataset points'%(title))
  if subtitle: ax1.set_title('%s\n%s'%(ax1.get_title(),subtitle))
  ax1.set_xlabel(fnames[0], labelpad=2)
  ax1.set_ylabel(fnames[1], labelpad=2)

  ax1.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=cm_bright)
  ax1.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6)

  ax1.set_xlim(xx.min(), xx.max())
  ax1.set_ylim(yy.min(), yy.max())
  ax1.set_xticks(())
  ax1.set_yticks(())

  ### Axes 2: decision boundary.
  ax2 = mpl_pyplot.subplot(1, 2, 2)
  ax2.set_title('%s: decision boundary'%(title))
  ax2.set_xlabel(fnames[0], labelpad=2)
  ax2.set_ylabel(fnames[1], labelpad=2)

  #Assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
  if hasattr(clf,"decision_function"):
    ax2.set_title('%s: decision_function contours'%(title))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  else:
    ax2.set_title('%s: prediction contours'%(title))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  ax2.contourf(xx, yy, Z, cmap=cm_contour, alpha=.8)

  ax2.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=cm_bright)
  ax2.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6)

  ax2.set_xlim(xx.min(), xx.max())
  ax2.set_ylim(yy.min(), yy.max())
  ax2.set_xticks(())
  ax2.set_yticks(())
  score = clf.score(X_test, y_test)
  ax2.text(xx.max()-.3, yy.min()+.3, ('%.2f'%score).lstrip('0'), size=15, horizontalalignment='right')

  fig.subplots_adjust(left=.02, right=.98)

  if ofile:
    fig.savefig(ofile)

  return fig

##############################################################################
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='SciKit-Learn utility')
  ops = ['Vectorize', 'GenerateRandomDataset', 'CheckCSV', 'SplitCSV']
  parser.add_argument("op", choices=ops, help='operation')
  parser.add_argument("--i", dest="ifile", help="input, CSV with N_features+1 cols, last col class labels")
  parser.add_argument("--noheader", action="store_true", help="CSV lacks header")
  parser.add_argument("--delim", default='\t', help="TSV/CSV delimiter")
  parser.add_argument("--defval", type=float, default=0.0, help="default numeric value")
  parser.add_argument("--coltags", help="column names, comma-separated")
  parser.add_argument("--tsv", action="store_true", help="delim is tab")
  parser.add_argument("--o", dest="ofile", help="output (CSV)")
  parser.add_argument("--osplit", dest="ofile_split", help="2nd output file (CSV)")
  parser.add_argument("--nclass", type=int, default=2, help="N classes")
  parser.add_argument("--nfeat", type=int, help="N features")
  parser.add_argument("--nsamp", type=int, help="N samples")
  parser.add_argument("--split_pct", type=float, default=10.0, help="pct of input randomly split into testset")
  parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0)
  args = parser.parse_args()

  logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

  if args.ifile:
    fin = open(args.ifile)
    if not fin: parser.error('cannot open ifile: %s'%args.ifile)

  fout = open(args.ofile, "w") if args.ofile else sys.stdout

  delim = '\t' if args.tsv else args.delim

  #csv.register_dialect("skl", strict=True, delimiter=delim, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

  mpl.use('Agg')

  if args.op == "Vectorize":
    if not args.ifile: parser.error('input file required.')
    #VectorizeCSV(fin, fout, args.coltags, csv.get_dialect("skl"))
    VectorizeCSV(fin, fout, args.delim, args.coltags)

  elif args.op == "GenerateRandomDataset":
    GenerateRandomDataset(args.nclass, args.nsamp, args.nfeat, fout)

  elif args.op == "CheckCSV":
    #CheckCSV(fin, csv.get_dialect("skl"))
    CheckCSV(fin, args.delim)

  elif args.op == "SplitCSV":
    if args.ofile_split:
      fout_split = open(args.ofile_split, "w")
    else:
      parser.error('--osplit required.')
    #SplitCSV(fin, fout, fout_split, args.split_pct, csv.get_dialect("skl"))
    SplitCSV(fin, fout, args.delim, fout_split, args.split_pct)
