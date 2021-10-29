#!/usr/bin/env python3
#############################################################################
### scikit_classify.py - Scikit-Learn classifier methods and utilities
### 
### http://scikit-learn.org/
### 
### Dataset: (X,y)
###   X = float array, N samples * n features
###   y = integer labels, N * (1 or 0)
#############################################################################
### RBF SVM parameters
### http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
### Intuitively, the gamma parameter defines how far the influence of a single
### training example reaches, with low values meaning ‘far’ and high values meaning
### ‘close’. The gamma parameters can be seen as the inverse of the radius of influence
### of samples selected by the model as support vectors.
### The C parameter trades off misclassification of training examples against simplicity
### of the decision surface. A low C makes the decision surface smooth, while a high C
### aims at classifying all training examples correctly by giving the model freedom to
### select more samples as support vectors.
#############################################################################
### X, y = skl_make_classification(n_samples=10, n_classes=2, n_features=3, n_informative=2, n_redundant=0)
### 
### X
### array([[ 0.40224457, -0.70270726, -1.1188435 ],
###        [-0.28613246, -0.60869432, -0.11959901],
###        [ 0.0163951 ,  1.09353508,  2.4999765 ],
###        [ 0.65721875,  1.20139004, -1.07410393],
###        [-0.28591664,  2.40319221, -3.77212832],
###        [-0.58094337, -1.18957265,  1.08500196],
###        [-1.15045791,  1.17969191, -2.21232356],
###        [ 1.35085948,  1.32776218,  1.85506999],
###        [ 0.43424904, -0.54260143, -0.26590429],
###        [ 0.10922903,  1.22767415,  2.23781926]])
### 
### X.shape
### (10, 3)
### 
### y
### array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])
#############################################################################
import sys,os,re,argparse,logging
import random,time,csv

import numpy as np
import matplotlib as mpl #additional imports follow
import pandas as pd

import sklearn.metrics 
import sklearn.model_selection
from sklearn.datasets import make_classification as skl_make_classification
from sklearn.preprocessing import StandardScaler as skl_StandardScaler
#
from sklearn.ensemble import RandomForestClassifier as skl_RandomForestClassifier, AdaBoostClassifier as skl_AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as skl_KNeighborsClassifier
from sklearn.svm import SVC as skl_SVC
from sklearn.tree import DecisionTreeClassifier as skl_DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB as skl_GaussianNB
from sklearn.neural_network import BernoulliRBM as skl_BernoulliRBM, MLPClassifier as skl_MLPClassifier
from sklearn.decomposition import PCA as skl_PCA

import sklearn_utils

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
def ClassifierFactory(args):
  alg =args.alg.upper()
  clf=None;
  if alg=='AB':
    clf = skl_AdaBoostClassifier(algorithm='SAMME.R')
  elif alg=='DT':
    clf = skl_DecisionTreeClassifier(criterion='gini',max_depth=None,max_features=None)
  elif alg=='KNN':
    clf = skl_KNeighborsClassifier(n_neighbors=4,algorithm='auto',metric='minkowski',p=2)
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
if __name__=='__main__':
  PROG=os.path.basename(sys.argv[0])
  epilog='''\
Classifier algorithms:
  AB = AdaBoost,
  DT = Decision Tree,
  KNN = K-Nearest Neighbors,
  LDA = Linear Discriminant Analysis,
  MLP = Multi-layer Perceptron (Neural Network),
  NB = Gaussian Naive Bayes,
  RF = Random Forest,
  SVM = Support Vector Machine
'''
  svm_kernels = ['linear', 'rbf', 'sigmoid'] # 'poly' not working?
  parser = argparse.ArgumentParser(description='SciKit-Learn classifier utility', epilog=epilog)
  ops = ['train', 'train_and_test', 'crossvalidate', 'demo']
  parser.add_argument("op", choices=ops, help='operation')
  parser.add_argument("--itrain", dest="ifile_train", help="input, training, CSV with N_features+1 cols, one endpoint col")
  parser.add_argument("--itest", dest="ifile_test", help="input, test, CSV with N_features cols")
  parser.add_argument("--i", dest="ifile", help="input, for both train and test")
  parser.add_argument("--algorithm", dest="alg", default="RF", help="(KNN|MLP|NB|RF|etc.)")
  parser.add_argument("--show_plot", action="store_true", help="interactive display")
  parser.add_argument("--title", help="for plot etc.")
  parser.add_argument("--subtitle", help="for plot etc.")
  parser.add_argument("--delim", default=', ', help="CSV delimiter")
  parser.add_argument("--eptag", help="endpoint column (default=last)")
  parser.add_argument("--ignore_tags", help="feature columns to ignore (comma-separated)")
  parser.add_argument("--tsv", action="store_true", help="delim is tab")
  parser.add_argument("--nclass", type=int, default=2, help="N classes")
  parser.add_argument("--nfeat", type=int, help="N features")
  parser.add_argument("--nsamp", type=int, help="N samples")
  parser.add_argument("--nn_layers", type=int, default=100, help="NN hidden layers")
  parser.add_argument("--nn_max_iter", type=int, default=500, help="NN max iterations")
  parser.add_argument("--svm_kernel", choices=svm_kernels, default='rbf', help="SVM kernel")
  parser.add_argument("--svm_cparam", type=float, default=1.0, help="SVM C-parameter")
  parser.add_argument("--svm_gamma", default='auto', help="SVM gamma-parameter (non-linear kernel only)")
  parser.add_argument("--cv_folds", type=int, default=5, help="cross-validation folds")
  parser.add_argument("--classnames", help="correspond with 0, 1, 2...")
  parser.add_argument("--o", dest="ofile", help="output with predictions (CSV)")
  parser.add_argument("--oplot", dest="ofile_plot", help="output plot (PNG)")
  parser.add_argument("--plot_width", type=int, default=7, help="width in inches")
  parser.add_argument("--plot_height", type=int, default=5, help="height in inches")
  parser.add_argument("--plot_dpi", type=int, default=100, help="dots per inch")
  parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0)
  args = parser.parse_args()

  logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

  fin_train=None; fin_test=None;
  if args.ifile_train:
    fin_train = open(args.ifile_train)
  elif args.ifile:
    fin_train = open(args.ifile)

  if args.ifile_test:
    fin_test = open(args.ifile_test)
  elif args.ifile:
    fin_test = open(args.ifile)

  if args.ofile:
    fout=fout=open(args.ofile, "w")
  else:
    fout=None

  t0=time.time()

  if not args.show_plot:
    mpl.use('Agg')
  import matplotlib.pyplot as mpl_pyplot
  import matplotlib.colors as mpl_colors #ListedColormap

  logging.debug(f"MATPLOTLIB_BACKEND: {mpl.get_backend()}")

  clf = ClassifierFactory(args)

  if not clf:
    parser.error('ERROR: Failed to instantiate algorithm "%s"'%args.alg)

  title = args.title if args.title else re.sub(r"^.*'.*\.(\w*)'.*$", r'\1', str(type(clf)))
  delim = '\t' if args.tsv else args.delim

  csv.register_dialect("skl", strict=True, delimiter=delim, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

  params = clf.get_params()
  logging.debug('Classifier type: {}'.format(re.sub(r"^.*'(.*)'.*$", r'\1', str(type(clf)))))
  for key in sorted(params.keys()):
    logging.debug(f"Classifier parameters: {key:>18}: {params[key]}")

  if args.op=="demo":
    Demo(clf, args.nclass, args.nfeat, args.nsamp, args.show_plot, args.ofile_plot)

  X, y = None,None

  if args.op in ("train", "train_and_test", "crossvalidate"):
    if not fin_train: parser.error('ERROR: input training file required.')
    X, y, ftags, eptag = sklearn_utils.ReadDataset(fin_train, eptag=args.eptag, ignore_tags=args.ignore_tags, csvdialect=csv.get_dialect("skl"))
    clf.fit(X,y)

  if args.op == "crossvalidate":
    CrossValidate(clf, X, y, args.cv_folds)

  if args.op == "train_and_test":
    if X is None or y is None: parser.error('ERROR: trained model required.')
    if not fin_test: parser.error('ERROR: input test file required.')
    X_test,y_test,ftags,eptag = sklearn_utils.ReadDataset(fin_test, eptag=args.eptag, ignore_tags=args.ignore_tags, csvdialect=csv.get_dialect("skl"))
    sklearn_utils.TestClassifier(clf, X_test, y_test, 'testset', fout)

    if args.show_plot or args.ofile_plot:
      if X.shape[1]>2:
        cnames = re.split(r'\s*,\s*', args.classnames.strip()) if args.classnames else None
        sklearn_utils.PlotPCA(clf, X, y, X_test, y_test, cnames, args.eptag, title, args.subtitle, args.plot_width, args.plot_height, args.plot_dpi, args.ofile_plot)
      else:
        PlotClassifier(clf, X, y, X_test, y_test, None, args.eptag, title, args.subtitle, args.ofile_plot)

      if args.show_plot:
        mpl_pyplot.show()

  if args.ofile:
    fout.close()

  logging.info("Elapsed time: "+(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

