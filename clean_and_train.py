import pandas as pd
import string

fake_df=pd.read_csv('data\\Fake.csv')
real_df=pd.read_csv('data\\True.csv')

fake_df['label']='Fake'
real_df['label']='Real'

df_fake_news=pd.concat([fake_df,real_df], axis=0).reset_index(drop=True)

df_fake_news=df_fake_news[['title', 'label']].dropna()

def clean_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    return text

df_fake_news['clean_title']=df_fake_news['title'].apply(clean_text)
df_fake_news.to_csv('data/fake_news_clean.csv', index=False)
print("Saved dataset")


df_topic=pd.read_csv('data\\ag_news_dataset.csv',header=0)

df_topic.columns=['class_index','title','description']
df_topic['text'] = df_topic['title'] + ' ' + df_topic['description']

topic_labels = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

df_topic['label'] = df_topic['class_index'].map(topic_labels)

df_topic['clean_text'] = df_topic['text'].apply(clean_text)

df_topic[['clean_text', 'label']].to_csv('data/topic_news_clean.csv', index=False)

print("saved succesfully")