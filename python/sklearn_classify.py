#!/usr/bin/env python3
###
"""
Scikit-Learn classifier methods and utilities
https://scikit-learn.org/
"""
#############################################################################
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
import random,time
#import csv #replace with pandas

import numpy as np
import matplotlib as mpl #additional imports follow
import pandas as pd

import sklearn

import sklearn_utils

##############################################################################
if __name__=='__main__':
  ALGOS = {"AB":"AdaBoost", "DT":"Decision Tree", "KNN":"K-Nearest Neighbors", "LDA":"Linear Discriminant Analysis", "MLP":"Multi-layer Perceptron (Neural Network)", "NB":"Gaussian Naive Bayes", "RF":"Random Forest", "SVM":"Support Vector Machine"}
  SVM_KERNELS = ['linear', 'rbf', 'sigmoid'] # 'poly' not working?
  EPILOG="""\
Classifier algorithms: {}
""".format("; ".join([f"{key}:{val}" for key,val in ALGOS.items()]))
  parser = argparse.ArgumentParser(description='SciKit-Learn classifier utility', epilog=EPILOG)
  parser.add_argument("--demo", action="store_true")
  parser.add_argument("--trainOnly", action="store_true", help="Default is full train-crossvalidate-test, workflow.")
  parser.add_argument("--itrain", dest="ifile_train", help="input, training, CSV with N_features+1 cols, one endpoint col")
  parser.add_argument("--itest", dest="ifile_test", help="input, test, CSV with N_features cols")
  parser.add_argument("--i", dest="ifile", help="input, for both train and test")
  parser.add_argument("--algorithm", dest="algo", choices=ALGOS.keys(), default="KNN", help="classifier algorithm")
  parser.add_argument("--show_plot", action="store_true", help="interactive display")
  parser.add_argument("--title", help="for plot etc.")
  parser.add_argument("--subtitle", help="for plot etc.")
  parser.add_argument("--delim", default="\t", help="CSV delimiter")
  parser.add_argument("--eptag", help="endpoint column (default=last)")
  parser.add_argument("--ignore_tags", help="feature columns to ignore (comma-separated)")
  parser.add_argument("--tsv", action="store_true", help="delim is tab")
  parser.add_argument("--nclass", type=int, default=2, help="N classes")
  parser.add_argument("--nfeat", type=int, help="N features")
  parser.add_argument("--nsamp", type=int, help="N samples")
  parser.add_argument("--nn_layers", type=int, default=100, help="NN hidden layers")
  parser.add_argument("--nn_max_iter", type=int, default=500, help="NN max iterations")
  parser.add_argument("--svm_kernel", choices=SVM_KERNELS, default="rbf", help="SVM kernel")
  parser.add_argument("--svm_cparam", type=float, default=1.0, help="SVM C-parameter")
  parser.add_argument("--svm_gamma", type=float, help="SVM gamma-parameter (non-linear kernel only)")
  parser.add_argument("--cv_folds", type=int, default=5, help="cross-validation folds")
  parser.add_argument("--classnames", help="correspond with 0, 1, 2...")
  parser.add_argument("--o", dest="ofile", help="output with predictions (CSV)")
  parser.add_argument("--oplot", dest="ofile_plot", help="output plot (PNG)")
  parser.add_argument("--plot_width", type=int, default=7, help="width in inches")
  parser.add_argument("--plot_height", type=int, default=5, help="height in inches")
  parser.add_argument("--plot_dpi", type=int, default=100, help="dots per inch")
  parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0)
  args = parser.parse_args()

  logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 or args.op=="demo" else logging.INFO))

  fin_train = open(args.ifile_train) if args.ifile_train else open(args.ifile) if args.ifile else None
  fin_test = open(args.ifile_test) if args.ifile_test else open(args.ifile) if args.ifile else None

  fout = open(args.ofile, "w") if args.ofile else None

  delim = "\t" if args.tsv else args.delim

  t0=time.time()

  logging.info(f"Scikit-learn version: {sklearn.__version__}")

  if not args.show_plot:
    mpl.use('Agg')
  import matplotlib.pyplot as mpl_pyplot
  import matplotlib.colors as mpl_colors #ListedColormap

  logging.debug(f"MATPLOTLIB_BACKEND: {mpl.get_backend()}")

  if args.demo:
    for algo in ALGOS.keys():
      logging.info(f"=== {algo:>8}: {ALGOS[algo]}")
      clf = sklearn_utils.ClassifierFactory(algo, args)
      if clf is not None:
        sklearn_utils.Demo(clf, args.nclass, args.nfeat, args.nsamp, args.show_plot, args.ofile_plot)
    sys.exit()

  X, y = None,None
  if not fin_train: parser.error('Input training file required.')
  clf = sklearn_utils.ClassifierFactory(args.algo, args)
  X, y, ftags, eptag = sklearn_utils.ReadDataset(fin_train, args.delim, eptag=args.eptag, ignore_tags=args.ignore_tags)
  clf.fit(X,y)

  if not args.trainOnly:
    sklearn_utils.CrossValidate(clf, X, y, args.cv_folds)
    if X is None or y is None: parser.error('trained model required.')
    if not fin_test: parser.error('Input test file required.')
    X_test,y_test,ftags,eptag = sklearn_utils.ReadDataset(fin_test, args.delim, eptag=args.eptag, ignore_tags=args.ignore_tags)
    sklearn_utils.TestClassifier(clf, X_test, y_test, 'testset', fout)
    if args.show_plot or args.ofile_plot:
      title = args.title if args.title else re.sub(r"^.*'.*\.(\w*)'.*$", r'\1', str(type(clf)))
      if X.shape[1]>2:
        cnames = re.split(r'\s*,\s*', args.classnames.strip()) if args.classnames else None
        fig = sklearn_utils.PlotPCA(clf, X, y, X_test, y_test, cnames, args.eptag, title, args.subtitle, args.plot_width, args.plot_height, args.plot_dpi)
      else:
        fig = sklearn_utils.Plot2by2Classifier(clf, X, y, X_test, y_test, None, args.eptag, title, args.subtitle)
      if args.ofile_plot: fig.savefig(ofile_plot)
      if args.show_plot: mpl_pyplot.show()
    if args.ofile: fout.close()

  logging.info("Elapsed time: "+(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

