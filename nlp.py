#Basic NLP Imports

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
from scipy import sparse


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold,train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import log_loss

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score,recall_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

%matplotlib inline

#Elite NLP Imports
import pandas as pd
import numpy as np
import csv
import os
import random
import torch
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
import sklearn.model_selection as model_selection
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForMaskedLM
import time
import copy
from tqdm.notebook import tqdm
