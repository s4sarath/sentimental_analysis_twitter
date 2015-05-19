import pickle
import pandas as pd
import numpy as numpy
import sklearn

###### Witing objects to a pickle 
### Read Training data
train_data = pd.read_csv('/Users/sarathrnair/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv',header=None)
#### Add columns to the data
train_data.columns = ['polarity','id','date','query','user','data']
X_positive = train_data.loc[train_data['polarity'] == 4]
X_negative = train_data.loc[train_data['polarity'] == 0]

### Segmenting only 200000 rows consideing the memory size
X_pos_seg = X_positive.iloc[0:100000,:]
X_neg_seg = X_negative.iloc[0:100000,:]
X_train = pd.concat([X_pos_seg, X_neg_seg], axis = 0)
print X_train.shape

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords





pt = PorterStemmer()




stopwords = set(stopwords.words("english"))
at_remove = re.compile(ur'@[_\s]{0,2}[a-zA-Z\d_]+\s{1}')
extra_remove = re.compile(ur"[^a-zA-Z]")



def remove_at(text):
    twitter_preprocessed = []
    for sent in text:
        
        prep_sent = re.sub(at_remove,'',sent)
        prep_sent = re.sub(extra_remove,' ',prep_sent)
        twitter_preprocessed.append(prep_sent)
    return twitter_preprocessed  

def tokenize(text):
    
    tokenized_twitter = []
    for sent in text:
        
        tokens = nltk.word_tokenize(sent)
      
        final_tokens = [pt.stem(w.lower()) for w in tokens if not w in stopwords and  len(w)>1]
        
        tokenized_twitter.append(final_tokens)
    return tokenized_twitter    
        

#### Expect the training data as an iterator over sentences one by one 

def preprocess(data):
    
    twitter_preprocess = remove_at(data)  
    twitter_tokens = tokenize(twitter_preprocess)
    return twitter_preprocess, twitter_tokens



twitter_preprocess, twitter_tokens = preprocess(X_train.data)

### Converting tokens into sentences as input for CountVewctorizer

twitter_token_sent = [' '.join(sent) for sent in twitter_tokens]    
    
    




# fileobj = open('/Users/sarathrnair/Sarath_works/Cfloor/cfloor_flask/sklearn_models/train_features_4k.pkl','rb')
# train_features = pickle.load(fileobj )
# fileobj.close()



import sklearn
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer( )
train_data_features = count_vect.fit_transform(twitter_token_sent)
train_features = train_data_features.toarray()

### Feature matrix shape
print "Shape of train feature matrix", train_features.shape


### Concatanating data with polarity

X = pd.DataFrame(train_features)
X = pd.concat([X,X_train.polarity],axis=1)

# fileobj = open('/Users/sarathrnair/Sarath_works/Cfloor/cfloor_flask/sklearn_models/test_dataframe_2k.pkl','wb')
# pickle.dump(X, fileobj)

#### Imporing models and craeting test and train split
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix

X_Data = X
Y = X.polarity
X = X.drop(['polarity','id'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)


rf = RF()
rf_model = rf.fit(X_train, Y_train)
predicted = rf_model.predict(X_test)
print 'Accuracy', np.mean(predicted == Y_test)


from sklearn.metrics import classification_report
target_names = ['0','4']
print(classification_report(Y_test, predicted, target_names=target_names))


%matplotlib inline
import matplotlib.pyplot as plt
cm = confusion_matrix(Y_test, predicted)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plot_confusion_matrix
print cm



