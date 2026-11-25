import pandas as pd
train=pd.read_csv('train.csv')
# print(train.head())
# print(train.shape)
# print(train.columns)
# train.info()
# print(train.describe())
# print(train['model'].value_counts())
# print(train['model'].nunique())
# print(train.isnull().sum())
# print(train['id'].nunique())
# train['text_length']=train['text'].str.len()
# print(train.groupby('model')['text_length'].mean().sort_values(ascending=False))
# import matplotlib.pyplot as plt
# train['text_length']=train['text'].str.len()
# avg=train.groupby('model')['text_length'].mean()
# print(avg)
# we have used pandas+matlab we use this sytntax to create barchart
# avg.plot(kind='bar')
# plt.xlabel("LLM model")
# plt.ylabel("Average character")
# plt.title("Average text length by llm")
# # plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig("llm_bar_pic.png")
# plt.show()
# ----- the avg response length usually has similar lengths so we need to caluculate how the response spread out-------
# train.boxplot(column='text_length' ,by='model',figsize=(14,6) )
# plt.title("Text Length Distribution by LLM")
# plt.xlabel("LLM Model")
# plt.ylabel("Number of Characters")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
# features
# 1
train['word_count']=train['text'].str.split().str.len()
# print(train[['text','word_count']].head())
# 2
train['sentence_count']=(train['text'].str.count(r'[.!?]'))
# print(train[['text','sentence_count','word_count']].head())
# 3
avg_word=train.groupby('model')['word_count'].mean()
print(avg_word.sort_values(ascending=False))
# 4
train['avg_wordpersente']=(train['word_count']/train['sentence_count'].replace(0,float('nan')))
print(train[['word_count','sentence_count','avg_wordpersente']].head())
print(train.groupby('model')['avg_wordpersente'].mean())
print(train[train['sentence_count']==0][['text','word_count','sentence_count']])
print(train['avg_wordpersente'].isnull().sum())
# 5
train['punctuation_count']=train['text'].str.count(r"""[,;:!?$"'()\[\]{}\/\\&%\-]""")
print(train['punctuation_count'].head())
print(train.groupby('model')['punctuation_count'].mean())
# 6 unique words
train['uniquewords_count']=train['text'].apply(lambda text:len(set(text.split())))
print(train['uniquewords_count'].head())
print(train.groupby('model')['uniquewords_count'].mean().sort_values(ascending=False))
# 7
train['vocablary_count']=(train['uniquewords_count']/train['word_count'])
print(train[['text','word_count','vocablary_count']].head())
print(train.groupby('model')['vocablary_count'].mean().sort_values(ascending=False))
# 8
train['text_length']=train['text'].str.len()
train['avg_word_length']=train['text_length']/train['word_count']
print(train.groupby('model')['avg_word_length'].mean().sort_values(ascending=False))
# 9
train['upper_count']=train['text'].apply(lambda text:sum(1 for char in text if char.isupper()))
print(train.groupby('model')['upper_count'].mean().sort_values(ascending=False))
# 10
train['digits']=train['text'].apply(lambda x:sum(1 for digi in x if digi.isdigit()))
print(train.groupby('model')['digits'].mean().sort_values(ascending=False))
# 11
train['newline_count']=train['text'].str.count('\n')
print(train['text'].str.count('\n').sum())
# 12
train['ques_count']=train['text'].str.count(r'[?]')
print(train.groupby('model')['ques_count'].mean().sort_values(ascending=False))
# 13
train['excla_count']=train['text'].str.count(r'[!]')
print(train.groupby('model')['excla_count'].mean().sort_values(ascending=False))
# 14
train['para_count']=train['text'].str.count(r'\n\n')
print(train.groupby('model')['para_count'].mean().sort_values(ascending=False))
# 15
stopwords={
    'the', 'is', 'am', 'are', 'was', 'were',
    'a', 'an', 'in', 'on', 'of', 'to', 'and',
    'for', 'with', 'it', 'this', 'that', 'be',
    'you', 'your', 'i', 'we', 'they', 'he', 'she'
}
train['stopwords_ratio']=train['text'].apply(lambda x :sum(1 for swords in x.lower().split() if swords in stopwords)/len(x.split()))
print(train.groupby('model')['stopwords_ratio'].mean().sort_values(ascending=False))

print(train.columns.tolist())

print(train.isnull().sum())

print(train[train['avg_wordpersente'].isnull()][
    ['text', 'word_count', 'sentence_count', 'avg_wordpersente']
])

print(train.nunique())

train=train.drop(columns=['newline_count','para_count'])
print(train)
print(train.columns.to_list())

print(train['text'].duplicated().sum())

print(train['model'].value_counts())

# making model to ready to realize X and y
x=train.drop(columns=['id','prompt_id','text','model']) #it has[row0,row1,row2]
y=train['model']
print(x.shape)
print(y.shape)
print(train['prompt_id'].nunique())
print(train['prompt_id'].head())
print(train['prompt_id'].value_counts().head(100))

# ==================================
# DATA PREPARATION FOR MACHINE LEARNING
# ==================================

from sklearn.model_selection import GroupShuffleSplit
group_splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
split_result=group_splitter.split(x,y,groups=train['prompt_id'])
train_idx,valid_idx=next(split_result) #train_dx=[0,2] valid[1,3]
print(len(train_idx))
print(len(valid_idx))
x_train=x.iloc[train_idx]#row0,row2
x_val=x.iloc[valid_idx]#row1,row3
y_train = y.iloc[train_idx]
y_val = y.iloc[valid_idx]
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)