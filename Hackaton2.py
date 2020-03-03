#!/usr/bin/env python
# coding: utf-8

# - Tzvi Puchinsky ID:203195706
# - Alla Kitaieva ID:336382833
# - Or Shalit ID:204469027

# In[1]:


import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
from functools import reduce
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# Read the csv data files

# In[2]:


anime_csv = pd.read_csv('anime.csv')
rating_csv = pd.read_csv('rating.csv')


# ### Anime CSV File

# In[3]:


anime_csv.head()


# Removing the episodes column, as we decided that is not relevant

# In[4]:


anime_csv = anime_csv.drop(columns='episodes')
anime_csv.head()


# Remove rows with empty elements, example: rows with missing rating values

# In[5]:


anime_csv = anime_csv.dropna()


# Bining the rating column

# In[6]:


bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
anime_csv['rating'] = pd.cut(anime_csv['rating'], bins=bins, labels=labels)
anime_csv.head()


# #### Create dict for ids & anime name

# In[7]:


ids_names_dict = dict(zip(anime_csv.anime_id, anime_csv.name))


# ### Rating CSV file

# In[8]:


rating_csv.head()


# In[9]:


rating_csv.shape


# Removing all the rows with ratings -1

# In[10]:


rating_csv = rating_csv[rating_csv['rating'] != -1]
rating_csv.head()


# In[11]:


rating_csv.shape


# In[12]:


reader= Reader(rating_scale=(1, 9))
rating_data = Dataset.load_from_df(rating_csv[['user_id', 'anime_id','rating']], reader)


# In[13]:


benchmark = []
# Iterate over all algorithms
for algorithm in [BaselineOnly()]:
    # Perform cross validation
    results = cross_validate(algorithm, rating_data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    


# <b>Cross Validate</b><br/>
# 1. algo = defined as baseline only
# 2. rating_data = our data
# 3. Measures for evaluation = RMSE and MAE
# 4. cv = cross validation iterator. Passed 3 - 3 folds
# 5. Verbose = (False) dont print averges and standard deviatonis over all splits

# In[14]:


print('Using ALS')
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
algo = BaselineOnly(bsl_options=bsl_options)
cross_validate(algo, rating_data, measures=['RMSE', 'MAE'], cv=3, verbose=False)


# In[15]:


trainset, testset = train_test_split(rating_data, test_size=0.3)
algo = BaselineOnly(bsl_options=bsl_options)
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)
print(accuracy.rmse(predictions))
accuracy.mae(predictions)
print(accuracy.mae(predictions))


# In[16]:


def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0


# In[17]:


df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
df.head()


# In[18]:


best_predictions = df.sort_values(by='err')[:10]
best_predictions


# ## Recommendation function for user id

# In[19]:


def get_rec(userId, top=5):
    rec = df[df['uid'] == userId].sort_values(by='err')[:top]['iid']
    rec = [ids_names_dict[item] for item in rec]
    return rec


# In[20]:


get_rec(64142)


# In[21]:


defaultdict_list = list(rating_csv.user_id.unique())
defaultdict = {defaultdict_list:[] for defaultdict_list in defaultdict_list}


# In[22]:


def precision_recall_at_k(predictions, k=5, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


# In[23]:


precisions, recalls = precision_recall_at_k(predictions)


# In[24]:


precisions_total = reduce(lambda x, value: x * (1 if value == 0 else value),precisions.values(), 1)
print('Precision total: ' + str(precisions_total))


# In[25]:


recalls_total = reduce(lambda x, value: x * (1 if value == 0 else value),recalls.values(), 1)
print('Recall total: ' + str(recalls_total))


# ### Calculating F1 using sklearn f1_score

# In[26]:


y_true = list()
y_pred = list()


# In[27]:


for key in defaultdict.keys():
    if defaultdict[key]:  # Contatins empty list
        for item in defaultdict[key]:
            y_true.append(item[1])
            y_pred.append(round(item[0]))
        


# 'macro': <br/>Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 

# In[28]:


f1_score_ans =  f1_score(y_true, y_pred, average='macro')
f1_score_ans


# In[29]:


print('Summary:')
print('RMSE: 1.2119')
print('MAE:  0.9301')
print('F1 score: {0}'.format(f1_score_ans))


# RMSE (root mean squared error): a value of 0 (almost never achieved in practice) would indicate a perfect fit to the data. In general, a lower RMSE is better than a higher one. <br/>
# MAE (mean absolute error): same as RMSE <br/>
# F1 score: F1 = 2 * (precision * recall) / (precision + recall)

# # Feature Recommendation
# <b>Note</b>: This part uses cosine similiraty to recommend. This part does not contain evaluation.

# <b>Load fresh data set </b><br/>
# <b>Anime dataset</b>

# In[30]:


anime_dataset = pd.read_csv('anime.csv')
anime_dataset.head()


# <b>Load rating data set<b/>

# In[31]:


rating_dataset = pd.read_csv('rating.csv')
rating_dataset.head()


# Reducing the length of the genre type to the first type in the list. (We wanted to include all the types but due to time limit skipped this part)

# In[32]:


anime_dataset["genre"]= anime_dataset["genre"].str.split(",", n = 2, expand = True)


# In[33]:


anime_dataset.head()


# In[34]:


anime_dataset = anime_dataset.dropna() # removing empty lines


# Bining the rating values to round values 0-10

# In[35]:


bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
anime_dataset['rating'] = pd.cut(anime_dataset['rating'], bins=bins, labels=labels)
anime_dataset.head()


# In[36]:


anime_dataset = anime_dataset.drop('episodes', axis=1)
anime_dataset.head()


# Converting the values to list values to be able to sum the lists to one vector list

# In[37]:


anime_dataset['name'] = anime_dataset['name'].astype(str)
anime_dataset['genre'] = anime_dataset['genre'].astype(str)
anime_dataset['type'] = anime_dataset['type'].astype(str)
anime_dataset['name'] = anime_dataset['name'].apply(lambda x: [x])
anime_dataset['genre'] = anime_dataset['genre'].apply(lambda x: [x])
anime_dataset['type'] = anime_dataset['type'].apply(lambda x: [x])


# Creating vector list for cosine similiraty

# In[38]:


anime_dataset['vector'] = anime_dataset['name'] + anime_dataset['genre'] + anime_dataset['type']
anime_dataset['vector'] = anime_dataset['vector'].apply(lambda x: ' '.join(x))
anime_dataset['vector']


# ### Creating the cosine similiraty matrix

# In[39]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(anime_dataset['vector'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# <br> Convert the name column back to string from list</br>

# In[40]:


anime_dataset['name'] = anime_dataset['name'].apply(lambda x: ' '.join(x))
anime_dataset['name']


# In[41]:


titles = anime_dataset['name']
indices = pd.Series(anime_dataset.index, index=anime_dataset['name'])


# ## Recommendation Funciton
# Gets as input a title name string and returns list of similar titles based on cosine similiraty

# In[42]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# ## Example for recommendation function

# In[43]:


print("Title input: " + anime_dataset['name'][1])


# In[44]:


get_recommendations(anime_dataset['name'][1])


# ## Recommendation using mean rating values of user

# Load rating.csv data

# In[45]:


user = pd.read_csv('rating.csv')
user.head()


# Remove -1 rows

# In[46]:


user = user[user['rating'] != -1]
user.head()


# Calculate Mean Rating Per User [MRPU]

# In[47]:


MRPU = user.groupby(['user_id']).mean().reset_index()
MRPU['mean_rating'] = MRPU['rating']

MRPU.drop(['anime_id','rating'],axis=1, inplace=True)
MRPU.head()


# In[48]:


user = pd.merge(user,MRPU,on=['user_id','user_id'])
user.head()


# Dropping rows with anime ids that has rating below mean rating of the user

# In[49]:


user = user.drop(user[user.rating < user.mean_rating-1].index)
user.head()


# Now we can find the most rated anime for specific user
# <br/> Lets take an exmpale user ID = 3

# In[50]:


user_id = 3
user_rec = user[user['user_id']== user_id]
user_rec.head()


# In[51]:


# Reset indexing of the data Frame
user_rec.reset_index(inplace = True) 


# Find the index of the max rating value

# In[52]:


user_rec['rating'].idxmax()


# In[53]:


# Id of the anime with the max rating for specific user
user_rec.iloc[user_rec['rating'].idxmax()]['anime_id'].astype(int)


# In[54]:


# Using the id found we will use the cosine similiraty recommendation function
anime_name = anime_dataset[anime_dataset['anime_id']==199]['name'].astype(str)
list(anime_name)[0]


# In[55]:


# Recommend
get_recommendations(list(anime_name)[0])

