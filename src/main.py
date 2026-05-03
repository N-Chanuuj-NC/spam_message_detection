from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dataset
messages = [
'Win money now', 'Free entry in contest', 'Claim your prize', 'Limited time offer',
'Congratulations you won', 'Get cash now', 'Click this link to win', 'Exclusive deal for you','Lottery',

'Hello friend', 'How are you doing', 'Let us meet tomorrow', 'Are you available today',
'Call me later', 'Let us discuss project', 'Lunch tomorrow?', 'See you soon','Have Fun']

labels=[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]

# Vectorization
vec = TfidfVectorizer()
X = vec.fit_transform(messages)

# learns all unique words from the messages
print(vec.get_feature_names_out())
# converts each message into numerical vectors
print(X.toarray())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy",model.score(X_test,y_test))

# Test
test = ['Win cash now', 'Hello bro']
X_new = vec.transform(test)
print("Predictions:", model.predict(X_new))