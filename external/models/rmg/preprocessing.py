# CREDITS: https://github.com/wuch15/Reviews-Meet-Graphs/blob/main/RMG.ipynb

from json import *
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import random
import pandas as pd

nltk.download('punkt')


def dataframe_to_dict(df):
    ratings = df.set_index('user')[['item', 'label']].apply(lambda x: (x['item'], float(x['label'])), 1) \
        .groupby(level='user').agg(lambda x: dict(x.values)).to_dict()
    return ratings


dataset = 'Digital_Music'

with open(f'../../../data/{dataset}/{dataset}_5.json', 'r') as f:
    rawdata = [JSONDecoder().decode(x) for x in f.readlines()]

for i in range(len(rawdata)):
    rawdata[i]['text'] = [word_tokenize(x) for x in sent_tokenize(rawdata[i]['reviewText'].lower())]

word_dict = {'PADDING': [0, 999999]}
for i in rawdata:
    for k in i['text']:
        for j in k:
            if j in word_dict:
                word_dict[j][1] += 1
            else:
                word_dict[j] = [len(word_dict), 1]

word_dict_freq = {}
for x in word_dict:
    if word_dict[x][1] >= 10:
        word_dict_freq[x] = [len(word_dict_freq), word_dict[x][1]]
print(len(word_dict_freq), len(word_dict))

embdict={}
cnt=0
with open('../../../data/glove.840B.300d.txt','rb')as f:
    linenb=0
    while True:
        line=f.readline()
        if len(line)==0:
            break
        line = line.split()
        word=line[0].decode()
        linenb+=1
        if len(word) != 0:
            emb=[float(x) for x in line[1:]]
            if word in word_dict_freq:
                embdict[word]=emb
                if cnt%100==0:
                    print(cnt,linenb,word)
                cnt+=1


print(len(embdict),len(word_dict_freq))
print(len(word_dict_freq))


emb_mat=[0]*len(word_dict_freq)
xp=np.zeros(300,dtype='float32')

temp_emb=[]
for i in embdict.keys():
    emb_mat[word_dict_freq[i][0]]=np.array(embdict[i],dtype='float32')
    temp_emb.append(emb_mat[word_dict_freq[i][0]])
temp_emb=np.array(temp_emb,dtype='float32')

mu=np.mean(temp_emb, axis=0)
Sigma=np.cov(temp_emb.T)

norm=np.random.multivariate_normal(mu, Sigma, 1)

for i in range(len(emb_mat)):
    if type(emb_mat[i])==int:
        emb_mat[i]=np.reshape(norm, 300)
emb_mat[0]=np.zeros(300,dtype='float32')
emb_mat=np.array(emb_mat,dtype='float32')
print(emb_mat.shape)

np.save(f'../../../data/{dataset}/embed_vocabulary.npy', emb_mat)

uir_triples=[]
for i in rawdata:
    if i['reviewerID'] == 'unknown':
        print("unknown user id")
        continue
    if i['asin'] == 'unknown':
        print("unkown item id")
        continue
    try:
        temp = {}
        doc = []
        for y in i['text']:
            doc.append([word_dict_freq[x][0] for x in y if x in word_dict_freq])
        temp['text'] = doc
        temp['item'] = i['asin']
        temp['user'] = i['reviewerID']
        temp['label'] = i['overall']
        uir_triples.append(temp)
    except:
        continue

for i in range(len(uir_triples)):
    uir_triples[i]['id']=i

MAX_SENT_LENGTH = 10
MAX_SENTS = 5
MAX_REVIEW_USER = 15
MAX_REVIEW_ITEM = 20
MAX_NEIGHBOR = 20

data = pd.DataFrame(uir_triples)

uidList, iidList = data['user'].unique().tolist(), data['item'].unique().tolist()

user2id = dict((uid, i) for(i, uid) in enumerate(uidList))
item2id = dict((iid, i) for(i, iid) in enumerate(iidList))
uid = list(map(lambda x: user2id[x], data['user']))
iid = list(map(lambda x: item2id[x], data['item']))
data['user'] = uid
data['item'] = iid

train_np=np.load(f'../../../data/{dataset}/Train.npy')
val_np=np.load(f'../../../data/{dataset}/Val.npy')
test_np=np.load(f'../../../data/{dataset}/Test.npy')

train_uir = []
val_uir = []
test_uir = []

for row in train_np:
    train_uir.append({
        'text': data[(data['user'] == row[0]) & (data['item'] == row[1])]['text'].values,
        'item': row[1],
        'user': row[0],
        'id': data[(data['user'] == row[0]) & (data['item'] == row[1])]['id'].values[0],
        'label': data[(data['user'] == row[0]) & (data['item'] == row[1])]['label'].values[0]
    })

for row in val_np:
    val_uir.append({
        'text': data[(data['user'] == row[0]) & (data['item'] == row[1])]['text'].values,
        'item': row[1],
        'user': row[0],
        'id': data[(data['user'] == row[0]) & (data['item'] == row[1])]['id'].values[0],
        'label': data[(data['user'] == row[0]) & (data['item'] == row[1])]['label'].values[0]
    })

for row in test_np:
    test_uir.append({
        'text': data[(data['user'] == row[0]) & (data['item'] == row[1])]['text'].values,
        'item': row[1],
        'user': row[0],
        'id': data[(data['user'] == row[0]) & (data['item'] == row[1])]['id'].values[0],
        'label': data[(data['user'] == row[0]) & (data['item'] == row[1])]['label'].values[0]
    })

train_dict = dataframe_to_dict(pd.DataFrame(train_uir))
users = list(train_dict.keys())
items = list({k for a in train_dict.values() for k in a.keys()})

private_to_public_users = {p: u for p, u in enumerate(users)}
public_to_private_users = {v: k for k, v in private_to_public_users.items()}
private_to_public_items = {p: i for p, i in enumerate(items)}
public_to_private_items = {v: k for k, v in private_to_public_items.items()}

item_review_id = {}
user_review_id = {}

for i in train_uir:
    if public_to_private_items[i['item']] in item_review_id:

        item_review_id[public_to_private_items[i['item']]].append(i['id'])
    else:
        item_review_id[public_to_private_items[i['item']]] = [i['id']]

    if public_to_private_users[i['user']] in user_review_id:

        user_review_id[public_to_private_users[i['user']]].append(i['id'])
    else:
        user_review_id[public_to_private_users[i['user']]] = [i['id']]

all_user_texts = []
for i in user_review_id:
    pad_docs = []
    for j in user_review_id[i][:MAX_REVIEW_USER]:
        sents = [x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        pad_sents = [x + (MAX_SENT_LENGTH - len(x)) * [0] for x in sents]
        pad_docs.append(pad_sents + [[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(pad_sents)))
    all_user_texts.append(pad_docs + [[[0] * MAX_SENT_LENGTH] * MAX_SENTS] * (MAX_REVIEW_USER - len(pad_docs)))

all_item_texts = []
for i in item_review_id:
    pad_docs = []
    for j in item_review_id[i][:MAX_REVIEW_ITEM]:
        sents = [x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        pad_sents = [x + (MAX_SENT_LENGTH - len(x)) * [0] for x in sents]
        pad_docs.append(pad_sents + [[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(pad_sents)))
    all_item_texts.append(pad_docs + [[[0] * MAX_SENT_LENGTH] * MAX_SENTS] * (MAX_REVIEW_ITEM - len(pad_docs)))

all_user_texts = np.array(all_user_texts, dtype='int32')
all_item_texts = np.array(all_item_texts, dtype='int32')

np.save(f'../../../data/{dataset}/all_user_texts.npy', all_user_texts)
np.save(f'../../../data/{dataset}/all_item_texts.npy', all_item_texts)

item_to_user_id = {}

for i in train_uir:
    if public_to_private_items[i['item']] in item_to_user_id:
        item_to_user_id[public_to_private_items[i['item']]].append(i['user'])
    else:
        item_to_user_id[public_to_private_items[i['item']]] = [i['user']]

user_to_item_id = {}

for i in train_uir:
    if public_to_private_users[i['user']] in user_to_item_id:
        user_to_item_id[public_to_private_users[i['user']]].append(i['item'])
    else:
        user_to_item_id[public_to_private_users[i['user']]] = [i['item']]

user_to_item_to_user = []
user_to_item = []
for i in user_to_item_id:
    ids = []

    ui_ids = user_to_item_id[i][:MAX_NEIGHBOR]

    for j in user_to_item_id[i]:
        randids = random.sample(item_to_user_id[j], min(MAX_NEIGHBOR, len(item_to_user_id[j])))

        ids.append(randids + [len(user_to_item_id) + 1] * (MAX_NEIGHBOR - len(randids)))
    ids = ids[:MAX_NEIGHBOR]
    user_to_item_to_user.append(ids + [[len(user_to_item_id) + 1] * MAX_NEIGHBOR] * (MAX_NEIGHBOR - len(ids)))

    user_to_item.append(ui_ids + [len(item_to_user_id) + 1] * (MAX_NEIGHBOR - len(ui_ids)))

user_to_item_to_user = np.array(user_to_item_to_user, dtype='int32')

user_to_item = np.array(user_to_item, dtype='int32')

np.save(f'../../../data/{dataset}/user_to_item_to_user.npy', user_to_item_to_user)
np.save(f'../../../data/{dataset}/user_to_item.npy', user_to_item)

item_to_user_to_item = []
item_to_user = []
for i in item_to_user_id:
    ids = []

    iu_ids = item_to_user_id[i][:MAX_NEIGHBOR]

    for j in item_to_user_id[i]:
        randids = random.sample(user_to_item_id[j], min(MAX_NEIGHBOR, len(user_to_item_id[j])))

        ids.append(randids + [len(item_to_user_id) + 1] * (MAX_NEIGHBOR - len(randids)))
    ids = ids[:MAX_NEIGHBOR]
    item_to_user_to_item.append(ids + [[len(item_to_user_id) + 1] * MAX_NEIGHBOR] * (MAX_NEIGHBOR - len(ids)))

    item_to_user.append(iu_ids + [len(user_to_item_id) + 1] * (MAX_NEIGHBOR - len(iu_ids)))

item_to_user_to_item = np.array(item_to_user_to_item, dtype='int32')
item_to_user = np.array(item_to_user, dtype='int32')

np.save(f'../../../data/{dataset}/item_to_user_to_item.npy', item_to_user_to_item)
np.save(f'../../../data/{dataset}/item_to_user.npy', item_to_user)
