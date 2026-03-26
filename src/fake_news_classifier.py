
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import logging
import re
import argparse


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from helper import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

logging.basicConfig(filename="message.log",
                    format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

logging.info("Downloading dataset via kagglehub...")
#LIAR dataset
path = kagglehub.dataset_download("mahdimashayekhi/fake-news-detection-dataset")
df = pd.read_csv(path+'/fake_news_dataset.csv')

logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
logging.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================

logging.debug(df.isnull().sum())
logging.debug(df['source'].mode())

df.fillna({'source': 'Daily News'}, inplace=True)
df.fillna({'author': 'Michael Smith'}, inplace=True)
logging.debug(df.isnull().sum())
logging.debug(df.duplicated().sum())

nltk.download('stopwords')
logging.debug(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    return ' '.join([ps.stem(w) for w in text if w not in stop_words])

logging.info("Preprocessing text (title + body concat)...")
# Concatenating title and body
df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(preprocess)

# =============================================================================
# 3. VECTORISATION
# =============================================================================

# adds bigrams — catches phrases like "breaking news" or "deep state" that 
# are strong fake-news signals and weak as unigrams.

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['content'])
y = df['label']

# =============================================================================
# 4. TRAIN / TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 5. TRAINING
# =============================================================================

logging.info("Training SVM (kernel=linear, C=1.0)...")
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# =============================================================================
# 6. PREDICTION & EVALUATION
# =============================================================================

y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred)

logging.info(f"Accuracy : {acc:.4f}")
logging.info(f"Macro F1 : {f1:.4f}")
logging.info(f"Classification report:\n{report}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm.classes_, yticklabels=svm.classes_)
plt.title('TF-IDF + SVM — Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('outputs/svm_confusion_matrix.png', dpi=150,bbox_inches='tight')
logging.info("Confusion matrix saved to outputs/svm_confusion_matrix.png")
if args.plot:
    plt.show()
plt.close()

feature_names = tfidf.get_feature_names_out()
coefs = svm.coef_.toarray()[0]
top_n = 20
top_fake = pd.Series(coefs, index=feature_names).nsmallest(top_n)
top_real = pd.Series(coefs, index=feature_names).nlargest(top_n)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_fake.plot(kind='barh', ax=axes[0], title='Top FAKE features', color='salmon')
top_real.plot(kind='barh', ax=axes[1], title='Top REAL features', color='steelblue')
plt.suptitle('TF-IDF + SVM — Feature Importance', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/svm_top_features.png', dpi=150,bbox_inches='tight')
logging.info("Confusion matrix saved to outputs/svm_top_features.png")
if args.plot:
    plt.show()
plt.close()

logging.info("SVM - TF-IDF run complete. Outputs written to outputs/")
