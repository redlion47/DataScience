#!/usr/bin/env python
# coding: utf-8

# ### Action Even Dataset Simulation
# Assumption:
# - 1,000 Room members, generating 30,000 action events over 2 month window (**Feb 1, 2020 to March 31, 2020**)
# - Number of actions a given user performs follows a *Power Law distribution*
# - Classes of the actions: message, post, interact, consume, search. (applying a normal distribution in a random sampler)
# - Random placement of actions from a normal distribution
# - Each action is associated with one interest and one domain
# - Every action event is associated with a language, interest and domain:
#   1. Each user is only associated with 1-3 languages (One language has to be an international language
#   2. 3-7 interests
#   3. Associated between 1-3 domains

# ### Steps
# 1. Load the csv files of languages, domain and interests
# 2. Generate unique ids and concatenate the datasets
# 3. Use pandas and numpy to generate a synthetic dataset with these columns:
#     `user_ids`,`timestamp`,`actions`,`languages(1-3)`,`interests(3-  7)`,`domains(1-3)`
# 4. Create a csv file

# In[1]:


# Import packages that will be used in the data generation script
import pandas as pd
from numpy import random as Rd
import random
import time as tm


# In[2]:


# Get data we will be using for the generation
Int_Lang =["English", "French", "Spanish", "Russian", "Arabic", "Chinese"]
actions = ["message", "post", "interact", "consume", "search"]
domain = ["Business services","Information technology","Manufacturing","Health care","Finance","Retail","Accounting and legal","Construction, repair and maintenance","Media","Restaurants, bars and food services"]
intsts = ["studying nature","being outdoors","rearranging furniture","decorating things","collecting things","listening to music","traveling","playing with children","solving problems","working with numbers","charity work","auto work/repair","gardening","meeting people","conserving natural resources","drawing, painting, sketching","studying art","organizing information/records","working with animals","being around animals","scientific research","studying the media","reading","analyzing movies","designing things","budgeting","joining public causes","talking about politics","hands-on activities","creating new things","learning how things work","philosophy","dissecting  an organism","dancing","ballet","bird watching","anticipating others’ needs","visiting the elderly","socializing","caring for the sick","giving advice","planning events","reading spiritual guides","building things","analyzing systems","studying languages","engaging in business","teaching others","exploring new places","supervising people","public speaking","using social media tools", "doing electrical work","studying stocks/investments","data processing","science fiction","photography","writing poetry and stories","programming computers","observing human behavior","church activities","solving crossword puzzles","studying artifacts","singing","playing team sports","playing individual sport","studying maps","selling things","analytical writing"]

lang_df = pd.read_csv("List-of-languages-csv.csv", header=None)
lang_df=lang_df.rename(columns={0:"Lang"})


# In[3]:


# This function creates and returns a dictionary of values randomly selected from our list
def dicGen(lst,name,lmt=1):
    randDict = {}
    values = Rd.choice(lst, size=lmt, replace=False)
    # Loop through the list created of random values to create a dict dynamically
    x=0
    while x<len(values):
        randDict[name + "_" +str(x)]= values[x]
        x+=1
        
    return randDict


# In[4]:


# This picks a random value from the list of a member's languages,interests and domains
def selector(df):
    df=df.dropna(axis=1)
#     print(df)
    variables = ["lang","interests","domain"]
    selection = {}
    for i in variables:
        var_lst = []
        # look for column names starting with each value in the list variable and create a list
        for x in df.columns:
            if x.startswith(i):
                var_lst.extend(df[x].to_list())
        selection[i] = random.choice(var_lst)
    return selection


# In[5]:


# This function returns a random time in a given range of time
def timeGen(startDate,endDate):
    timeFormat = '%Y-%m-%d  %H:%M:%S'
    prob = random.random()
    # Change start and end date values to timesyamp values
    startTime, endTime = tm.mktime(tm.strptime(startDate,timeFormat)), tm.mktime(tm.strptime(endDate,timeFormat))
    selectedTime = startTime + prob*(endTime-startTime)# Get a random date and time in our range
    rand_time = tm.strftime(timeFormat, tm.localtime(selectedTime))
    return rand_time


# In[6]:


userBio = []
# Create 1000 users with their interests, domain and language
for x in range(1000):
    # Each user details are all in a dict format
    details ={}
    details["user_id"] ="ALX_RM_"+ "0"*(4-len(str(x+1))) + str(x+1)# Create a user_id
    # Random Limits that determine the number of languages, interests and domain a user has
    lan_lmt = random.randint(1,3)
    int_lmt = random.randint(3,7)
    dom_lmt = random.randint(1,3)
    # Random Languages for user
    details.update(dicGen(lst=Int_Lang,name="langInt"))
    details.update(dicGen(lst=lang_df["Lang"],lmt=lan_lmt-1, name="lang"))
    # Random interests for user
    details.update(dicGen(lst=intsts,lmt=int_lmt, name="interests"))
    # Random domains for user
    details.update(dicGen(lst=domain,lmt=dom_lmt, name="domain"))
    # Append to the list of user bio data
    userBio.append(details)
# Create a dataframe of the users bio data
Bio_df = pd.DataFrame(userBio,columns=["user_id",
                                       "langInt_0",
                                       "Lang_0",
                                       "Lang_1",
                                       "interests_0",
                                       "interests_1",
                                       "interests_2",
                                       "interests_3",
                                       "interests_4",
                                       "interests_5",
                                       "interests_6",
                                       "domain_0",
                                       "domain_1",
                                       "domain_2"])
Bio_df.sample(10)


# In[7]:


genData =[]# A list of 30000 generated data of room members
for rand in range(30000):
    # Each interaction data is in dict format
    userdet = {}
    user=random.choice(Bio_df.user_id.to_list())#Randomly select a user from the biodata dataframe
    userdet["user_id"] = user
    userdet["date"]= timeGen("2020-02-01 00:00:00","2020-03-31 23:59:59")
    userdet["action"]= random.choice(actions)# Randomly select an action from the action list
    # Add the user's interaction data to list of interactions
    userdet.update(selector(Bio_df.loc[Bio_df["user_id"] ==user]))
    genData.append(userdet)
# Create a dataframe of all room members interactions    
RoomGen_df = pd.DataFrame(genData)
RoomGen_df.sample(10)


# In[8]:


# Save interaction data as a csv file.
RoomGen_df.to_csv(path_or_buf="Room_Members_Gen_Data.csv")


# In[ ]:




