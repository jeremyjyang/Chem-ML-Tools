#!/usr/bin/env python3
#############################################################################
### Scikit-learn utilities
### http://scikit-learn.org/
### 
### Dataset: (X,y)
###   X = float array, N samples * n features
###   y = integer labels, N * (1 or 0)
#############################################################################
import sys,os,time,re,argparse,logging,random,tempfile
import numpy as np
import pandas as pd

import matplotlib as mpl
#import matplotlib.pyplot as mpl_pyplot
#import matplotlib.colors as mpl_colors #ListedColormap

from PIL import Image

import sklearn
import sklearn.metrics
import sklearn.model_selection
from sklearn.datasets import make_classification as make_classification
from sklearn.feature_extraction import DictVectorizer as DictVectorizer
from sklearn.preprocessing import StandardScaler as StandardScaler
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier, AdaBoostClassifier as AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
from sklearn.svm import SVC as SVC
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB as GaussianNB
from sklearn.neural_network import BernoulliRBM as BernoulliRBM, MLPClassifier as MLPClassifier
from sklearn.decomposition import PCA as PCA
#
ALGOS = {"AB":"AdaBoost", "DT":"Decision Tree", "KNN":"K-Nearest Neighbors", "LDA":"Linear Discriminant Analysis", "MLP":"Multi-layer Perceptron (Neural Network)", "NB":"Gaussian Naive Bayes", "RF":"Random Forest", "SVM":"Support Vector Machine"}
SVM_KERNELS = ['linear', 'rbf', 'sigmoid'] # 'poly' not working?
#
##############################################################################
def Demo():
  imgfiles=[];
  for i,algo in enumerate(ALGOS.keys()):
    logging.info(f"=== {algo:>8}: {ALGOS[algo]}")
    try:
      clf = ClassifierFactory(algo, nn_layers=100, nn_max_iter=500, svm_gamma=None, svm_kernel="rbf", svm_cparam=1.0)
    except Exception as e:
      logging.error(f"Classifier instantiation failed ({algo}): {e}")
      clf=None;
    if clf is not None:
      fig = DemoClassifier(clf, nclass=2, nfeat=None, nsamp=None)
      f = tempfile.NamedTemporaryFile(prefix='sklearn_utils_', suffix='.png', delete=False)
      fig.savefig(f.name)
      imgfiles.append(f.name)
      logging.info(f"{i}. {f.name}")

  imgs = [Image.open(imgfile) for imgfile in imgfiles]
  img_all = Image.new('RGB', (imgs[0].width, len(imgs)*imgs[0].height))
  for i,img in enumerate(imgs):
    img_all.paste(img, (0, i*imgs[0].height))
  img_all.show()
  for imgfile in imgfiles: os.remove(imgfile)
  sys.exit()

##############################################################################
def DemoClassifier(clf, nclass, nfeat, nsamp):
  '''Demo classifier using randomly generated dataset for training and test.'''
  ### Example dataset: two interleaving half circles
  #X, y = sklearn.datasets.make_moons(noise=0.3, random_state=0)
  ### Example dataset: 
  #X, y = sklearn.datasets.load_diabetes() #regression
  #X, y = sklearn.datasets.load_breast_cancer() #classification

  nclass = nclass if nclass else 2
  nfeat = nfeat if nfeat else 2 #allows 2D plot
  nsamp = nsamp if nsamp else random.randint(50, 200)

  ###Generate random classification dataset
  X, y = make_classification(n_classes=nclass, n_samples=nsamp, n_features=nfeat, n_redundant=0, n_informative=2, random_state=random.randint(0, 100), n_clusters_per_class=1)

  # Preprocess dataset, split into training and test part
  X = StandardScaler().fit_transform(X)
  #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.4)
  X_train, X_test, y_train, y_test = SplitDataset(X, y, split_pct=25)

  ### Train the classifier:
  clf.fit(X,y)

  ### Test the classifier:
  TestClassifier(clf, X_train, y_train, 'train', None)
  TestClassifier(clf, X_test, y_test, 'test', None)
  TestClassifier(clf, X, y, 'train_and_test', None)

  fnames = [f"feature_{j+1:02d}" for j in range(nfeat)]
  epname = 'endpoint'

  title = re.sub(r"^.*'.*\.(.*)'.*$", r'\1', str(type(clf)))

  fig = PlotPCA(clf, X_train, y_train, X_test, y_test, fnames, epname, title, None, 7, 5, 100) if nfeat>2 else Plot2by2Classifier(clf, X_train, y_train, X_test, y_test, fnames, epname, title, None)
  return fig

##############################################################################
def ClassifierFactory(algo, nn_layers, nn_max_iter, svm_gamma, svm_kernel, svm_cparam):
  if algo=='AB':
    clf = AdaBoostClassifier(algorithm='SAMME.R')

  elif algo=='DT':
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None)

  elif algo=='KNN':
    clf = KNeighborsClassifier(n_neighbors=4, algorithm='auto', metric='minkowski', p=2)

  elif algo=='MLP':
    clf = MLPClassifier(hidden_layer_sizes=(nn_layers, ), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=nn_max_iter, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

  elif algo=='NB':
    clf = GaussianNB()

  elif algo=='RF':
    #AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_'
    #clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

  elif algo=='SVM':
    logging.debug(f"svm_gamma = '{svm_gamma}'")
    gamma = 'auto' if svm_gamma is None else svm_gamma
    clf = SVMClassifierFactory(kernel=svm_kernel, cparam=svm_cparam, gamma=gamma)

  elif algo=='LDA':
    logging.error(f"Not yet implemented: {algo}")
    clf=None;

  else:
    logging.error(f"Not yet implemented: {algo}")
    clf=None;

  if clf is not None:
    params = clf.get_params()
    logging.debug('Classifier type: {}'.format(re.sub(r"^.*'(.*)'.*$", r'\1', str(type(clf)))))
    for key in sorted(params.keys()):
      logging.debug(f"Classifier parameters: {key:>18}: {params[key]}")
  else:
    logging.error(f"Failed to instantiate algorithm '{algo}'")
  return clf

##############################################################################
def SVMClassifierFactory(kernel, cparam=1.0, gamma='auto'):
  clf = SVC(kernel=kernel, C=cparam, gamma=gamma)
  return clf

##############################################################################
def ReadDataset(fin, delim, eptag=None, ignore_tags=[]):
  '''Read from file.  Classification or regression.  All features and endpoint must be numeric.'''
  n_data=0; n_col=0; X=[]; y=[];
  df = pd.read_csv(fin, sep=delim)
  if not eptag:
    eptag = df.columns[-1]
  j_eptag = list(df.columns).index(eptag)
  ignore_tags = re.split(r'\s*,\s*', ignore_tags.strip()) if ignore_tags else []
  featuretags = list(df.columns)[:]
  featuretags.pop(j_eptag)
  for tag in ignore_tags:
    featuretags.remove(tag)
  logging.info(f"featuretags = {str(featuretags)}")
  logging.info(f"eptag = '{eptag}'; j_eptag = {j_eptag}")
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
  return (X, y, featuretags, eptag)

##############################################################################
#scipy.sparse.csr.csr_matrix
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
# '''What to do with missing values?  quoting=csv.QUOTE_NONNUMERIC means all non-quoted fields converted to float.'''
def VectorizeCSV(fin, fout, coltags):
  n_in=0;  n_out=0;
  df = pd.read_csv(fin, sep=delim)
  logging.debug(f"fieldnames: ({df.shape[1]}) {df.columns}")
  vec = DictVectorizer()
  rows_vectorized = vec.fit_transform(df.to_dict())
  logging.debug(f"vectorized fieldnames: {len(vec.get_feature_names())}")
  logging.debug(f"vectorized fieldnames = {str(vec.get_feature_names())}")
  logging.debug(f"type(rows_vectorized): {type(rows_vectorized)}")
  logging.debug(f"rows read: {n_in}")
  rv = rows_vectorized.todok()
  pd.DataFrame(rv).to_csv(fout, "\t", index=False)

##############################################################################
def CheckCSV(fin, delim):
  '''Ok for dataset?  All numeric?  Or needing vectorization.'''
  try:
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
def SplitCSV(fin, fout, delim, fout_split, split_pct):
  df = pd.read_csv(fin, sep=delim)
  logging.debug(f"fieldnames: ({df.shape[1]}) {df.columns}")
  df_train = df.sample(frac=split_pct/100, replace=False)
  df_test = df[list(set(df.index) - set(df_train.index))]
  logging.info(f"rows read: {df.shape[0]}")
  logging.info(f"rows out (train): {df_train.shape[0]} ({100.0*df_train.shape[0]/df.shape[0]:%.1f}%)") 
  logging.info(f"rows out (test): {df_test.shape[0]} ({100.0*df_test.shape[0]/df.shape[0]:.1f}%)")
  logging.info(f"rows out (TOTAL): {df.shape[0]}")
  return df.shape[0],df_train.shape[0],df_test.shape[0]

##############################################################################
def SplitDataset(X, y, split_pct):
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=split_pct/100)
  logging.debug(f"split: total: {X.shape[0]} ; train : {X_train.shape[0]} ({100.0*X_train.shape[0]/X.shape[0]:.1f}%) ; test: {X_test.shape[0]} ({100.0*X_test.shape[0]/X.shape[0]:.1f}%)")
  return X_train, X_test, y_train, y_test

##############################################################################
def GenerateRandomDataset(nclass, nsamp, nfeat, fout):
  X,y = make_classification( n_classes=nclass, n_samples=nsamp, n_features=nfeat, n_redundant=0, n_informative=2, random_state=random.randint(0, 100), n_clusters_per_class=1)
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
def PlotPCA(clf, X_train, y_train, X_test, y_test, cnames, epname, title, subtitle, width, height, dpi):
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

  logging.info(f"Test-predictions: T = {n_test_t:3d} ; TP = {n_test_tp:3d} ; TN = {n_test_tn:3d}")
  logging.info(f"Test-predictions: F = {n_test_f:3d} ; FP = {n_test_fp:3d} ; FN = {n_test_fn:3d}")

  pca_d = 2
  pca = PCA(n_components=pca_d)

  X = np.concatenate((X_train, X_test), axis=0)
  y = np.concatenate((y_train, y_test), axis=0)

  n_train = X_train.shape[0]
  n_test = X_test.shape[0]

  pca.fit(X)
  X_r = pca.transform(X)

  logging.debug(f"X.shape = {str(X.shape)}; X_r.shape = {str(X_r.shape)}")
  logging.info(f"PCA {X.shape[1]}D to {X_r.shape[1]}-component")
  logging.info(f"PCA explained variance ratio (1st 2 components): {str(pca.explained_variance_ratio_)}")

  fig = mpl.pyplot.figure(frameon=False, tight_layout=False)
  fig.set_size_inches((width, height))
  fig.set_dpi(dpi)
  ax = fig.gca()
  logging.debug(f"figure size: {fig.get_size_inches()} ; DPI: {fig.dpi}")
  colors = ["navy", "turquoise", "darkorange", "forestgreen"]
  lw=1;
  y_vals = list(set(y))
  y_vals.sort()
  ylabels = [cnames[int(y_val)] for y_val in y_vals] if cnames else [str(int(y_val)) for y_val in y_vals]

  X_r_train = X_r[:n_train]
  X_r_test = X_r[-n_test:]
  for y_val, color, ylabel in zip(y_vals, colors, ylabels):
    mpl.pyplot.scatter(X_r_train[y_train==y_val, 0], X_r_train[y_train==y_val, 1], color=color, marker=".", alpha=.8, lw=lw, label="train:"+ylabel) #Train
    #mpl.pyplot.scatter(X_r_test[y_test==y_val, 0], X_r_test[y_test==y_val, 1], color=color, marker="+", alpha=.8, lw=lw, label="test:"+ylabel) #Test

  X_r_test_fp = X_r_test[np.where(y_test_fp)]
  mpl.pyplot.scatter(X_r_test_fp[:, 0], X_r_test_fp[:, 1], color="red", marker="^", alpha=.5, lw=lw, label="test:FP") #red-^ FPs
  X_r_test_fn = X_r_test[np.where(y_test_fn)]
  mpl.pyplot.scatter(X_r_test_fn[:, 0], X_r_test_fn[:, 1], color="red", marker="v", alpha=.5, lw=lw, label="test:FN") #red-v FNs

  X_r_test_tp = X_r_test[np.where(y_test_tp)]
  mpl.pyplot.scatter(X_r_test_tp[:, 0], X_r_test_tp[:, 1], color="green", marker="^", alpha=.5, lw=lw, label="test:TP") #green-^ TPs
  X_r_test_tn = X_r_test[np.where(y_test_tn)]
  mpl.pyplot.scatter(X_r_test_tn[:, 0], X_r_test_tn[:, 1], color="green", marker="v", alpha=.5, lw=lw, label="test:TN") #green-v TNs

  mpl.pyplot.legend(loc="best", title=None, shadow=True, scatterpoints=1)
  mpl.pyplot.title(f"PCA ({pca_d}D) of {title}\n{subtitle if subtitle else ''}")
  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_xticklabels(ax.get_xticklabels(), size="small") #ticklabels go away?
  ax.set_yticklabels(ax.get_yticklabels(), size="small") #ticklabels go away?
  ax.annotate(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xycoords="axes fraction", xy=(0.0, -0.1), horizontalalignment="left", verticalalignment="top")
  ax.annotate(f"N_cases = {X.shape[0]}\nN_features = {X.shape[1]}", xycoords="axes fraction", xy=(0.8, 0.5), horizontalalignment="center", verticalalignment="top")
  ax.annotate(f"N_train = {n_train} ({n_train*100/X.shape[0]:.1f}%)\nN_test = {n_test} ({n_test*100/X.shape[0]:.1f}%)", xycoords="axes fraction", xy=(0.8, 0.35), horizontalalignment="center", verticalalignment="top")
  ax.annotate(f"TP = {n_test_tp}, TN = {n_test_tn}\nFP = {n_test_fp}, FN = {n_test_fn}", xycoords="axes fraction", xy=(0.8, 0.2), horizontalalignment="center", verticalalignment="top")
  ax.annotate(f"mean_accuracy = {clf.score(X_test, y_test)*100:.1f}%", xycoords="axes fraction", xy=(0.8, 0.05), horizontalalignment="center", verticalalignment="top")

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
  cm_contour = mpl.pyplot.cm.RdBu
  cm_bright = mpl.colors.ListedColormap(["#FF0000", "#0000FF"])

  #Assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
  #Decision function returns how far from decision surface, sign indicating which side.
  #if hasattr(clf, "decision_function"):
  #  #ax2.set_title(f"{title}: decision_function contours")
  #  #X_r = pca.transform(X)
  #  Z = clf.decision_function(X)
  #else:
  #  #ax2.set_title(f"{title}: prediction contours")
  #  Z = clf.predict_proba(X)

  # Put the result into a color plot
  #Z = Z.reshape(xplot_mesh.shape)

  ### FIX THIS...
  #logging.debug("Z.shape = %s"%(str(Z.shape)))
  #ax.contourf(xplot_mesh, yplot_mesh, Z, cmap=cm_contour, alpha=.8)
  return fig

##############################################################################
def CrossValidate(clf, X, y, cv_folds):
  cv_scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=cv_folds, verbose=bool(logging.getLogger().getEffectiveLevel()<=logging.DEBUG))
  logging.info(f"cv_scores={str(cv_scores)}")

##############################################################################
def Plot2by2Classifier(clf, X_train, y_train, X_test, y_test, fnames, epname, title, subtitle):
  '''Only for n_features = 2,  n_classes = 2.'''
  logging.debug("PLOT: "+title)
  mesh_h = .02  # mesh step size
  figsize = (12,8) #w,h in inches
  fig = mpl.pyplot.figure(figsize=figsize, dpi=100, frameon=False, tight_layout=False)

  X = np.concatenate((X_train,X_test),axis=0)
  x_min,x_max = X[:,0].min()-.5, X[:,0].max()+.5
  y_min,y_max = X[:,1].min()-.5, X[:,1].max()+.5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_h),
                          np.arange(y_min, y_max, mesh_h))
  my_colors = ['#FF0000', '#0000FF', '#00FF00', '#999999']

  #Need more colors for n_classes > 2 ?
  cm_contour = mpl.pyplot.cm.RdBu
  cm_bright = mpl.colors.ListedColormap(my_colors)

  ### Axes 1: dataset points only.
  ax1 = mpl.pyplot.subplot(1, 2, 1)
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
  ax2 = mpl.pyplot.subplot(1, 2, 2)
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

  mpl.use('Agg')

  if args.op == "Vectorize":
    if not args.ifile: parser.error('input file required.')
    VectorizeCSV(fin, fout, args.delim, args.coltags)

  elif args.op == "GenerateRandomDataset":
    GenerateRandomDataset(args.nclass, args.nsamp, args.nfeat, fout)

  elif args.op == "CheckCSV":
    CheckCSV(fin, args.delim)

  elif args.op == "SplitCSV":
    if args.ofile_split:
      fout_split = open(args.ofile_split, "w")
    else:
      parser.error('--osplit required.')
    SplitCSV(fin, fout, args.delim, fout_split, args.split_pct)
