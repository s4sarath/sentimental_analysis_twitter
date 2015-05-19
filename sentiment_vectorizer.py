### Its a part of vectorizer which I was unable to do in Ipython due to Python crash

### The source of the code is "sentimental_new in ~/Sarath_works




import pickle
fileobj = open('/Users/sarathrnair/Sarath_works/Cfloor/cfloor_flask/sklearn_models/twitter_token_sent.pkl','rb')
twitter_token_sent = pickle.load(fileobj)
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_data_features = count_vect.fit_transform(twitter_token_sent)
train_features = train_data_features.toarray()

fileobj2 = open('/Users/sarathrnair/Sarath_works/Cfloor/cfloor_flask/sklearn_models/train_features.pkl','wb')
pickle.dump(train_features, fileobj2 )

### Feature matrix shape
print "Shape of train feature matrix", train_features.shape



