#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Test 

Reference to the program requirement : https://docs.google.com/document/d/1F2n03zxx4YpsRf-2sXVcqnJmW8ToluiPJem-Hx6vNy8/edit?ts=5e7ca940

Note: 
    Introduction: 
        We would like to simulate a basic implementation of member activity in an environment such as The Room. In order to do this, 
        one source of data must be a stream of member activity in such an environment as they interact with each other, search for, 
        consume, and interact with existing content, and post new content. To enable this simulation, we would need to generate a 
        synthetic dataset consisting of user actions. Consider some number of members in The Room, interacting with other members 
        and content over some period of time. Each interaction will generate an "action" event. 
    
    Output: 
        A python program to generate an Action Event Dataset that meets the criteria described above. The end result will be a csv files which can be converted to a database table 
        containing columns: user_id, timestamp, action, language, interests and domain; and about 30,000 rows. 
        
        The output should satisfy the following constraints: 
            Each user can only be associated with between 1-3 languages (NB: 1 of the three has to be an international language)
            Each user can only be associated with between 3-7 interests 
            Each user only be associated with between 1-3 domains
        
        The script outputs two csv file: 
            1. Action Simulation Dataset - User profile data (user_ids and their languages)
            2. Action Simulation Dataset - Action data (user_ids, action, timestamp, interest and domain)
            
   Note: Matching a language to an action can happen during training. 

"""

import numpy as np 
import pandas as pd
import random

# import random  
from random import sample 

timedata = np.arange('2020-02-01', '2020-03-01', np.timedelta64(1,'s'), dtype='datetime64[s]')
actiondata = ['message', 'post', 'interact', 'consume', 'search']
domainsdata = ['Adventurous','Affable','Capable','Charming','Confident','Conscientious','Cultured','Dependable','Discreet','Dutiful','Encouraging','Exuberant','Fair','Fearless','Gregarious','Helpful','Humble','Imaginative','Impartial','Independent','Keen','Meticulous','Observant','Optimistic','Persistent','Precise','Reliable','Sociable','Trusting','Valiant']
useriddata = np.arange(start=1,stop=1001,step=1)

languageProffesional = ['English', 'French', 'Spanish','German', 'Portuguese']

languagedata = ['Acholi','Afrikaans','Akan','Albanian','Amharic','Ashante','Asl','Assyrian','Azerbaijani','Arabic','Azeri','Bajuni','Basque','Behdini','Belorussian','Bengali','Berber','Bosnian',
'Bravanese','Bulgarian','Burmese','Cakchiquel','Cambodian','Cantonese','Catalan','Chaldean','Chamorro','Chao-chow','Chavacano','Chin','Chuukese','Cree','Croatian','Czech','Dakota','Danish','Dari',
'Dinka','Diula','Dutch','Edo','Estonian','Ewe','Farsi','Finnish','Flemish','Fukienese','Fula','Fulani','Fuzhou','Ga','Gaddang','Gaelic',
'Gaelic-irish','Gaelic-scottish','Georgian','Gorani','Greek','Gujarati','Haitian Creole','Hakka','Hakka-chinese','Hausa','Hebrew','Hindi','Hmong','Hungarian','Ibanag','Ibo','Icelandic',
'Igbo','Ilocano','Indonesian','Inuktitut','Italian','Jakartanese','Japanese','Javanese','Kanjobal','Karen','Karenni','Kashmiri','Kazakh','Kikuyu','Kinyarwanda','Kirundi','Korean','Kosovan',
'Kotokoli','Krio','Kurdish','Kurmanji','Kyrgyz','Lakota','Laotian','Latvian','Lingala','Lithuanian','Luganda','Luo','Maay','Macedonian','Malay','Malayalam','Maltese','Mandingo','Mandarin','Mandinka',
'Marathi','Marshallese','Mien','Mina','Mirpuri','Mixteco','Moldavan','Mongolian','Montenegrin','Navajo','Neapolitan','Nepali','Nigerian Pidgin','Norwegian','Oromo','Pahari','Papago','Papiamento',
'Pashto','Patois','Polish','Portug.creole','Pothwari','Pulaar','Punjabi','Putian','Quichua','Romanian','Russian','Samoan','Serbian','Shanghainese','Shona','Sichuan',
'Sicilian','Sinhalese','Slovak','Somali','Sorani','Sundanese','Susu','Swahili','Swedish','Sylhetti','Tagalog','Taiwanese','Tajik','Tamil','Telugu','Thai','Tibetan',
'Tigre','Tigrinya','Toishanese','Tongan','Toucouleur','Trique','Tshiluba','Turkish','Twi','Ukrainian','Urdu','Uyghur','Uzbek','Vietnamese','Visayan','Welsh','Wolof','Yiddish','Yoruba','Yupik']

interestsdata = ['Accounting','Airlines/Aviation','Alternative Dispute Resolution','Alternative Medicine','Animation','Apparel/Fashion','Architecture/Planning',
'Arts/Crafts','Automotive','Aviation/Aerospace','Banking/Mortgage','Biotechnology/Greentech','Broadcast Media','Building Materials','Business Supplies/Equipment',
'Capital Markets/Hedge Fund/Private Equity','Chemicals','Civic/Social Organization','Civil Engineering','Commercial Real Estate','Computer Games','Computer Hardware',
'Computer Networking','Computer Software/Engineering','Computer/Network Security','Construction','Consumer Electronics','Consumer Goods','Consumer Services','Cosmetics',
'Dairy','Defense/Space','Design','E-Learning','Education Management','Electrical/Electronic Manufacturing','Entertainment/Movie Production','Environmental Services',
'Events Services','Executive Office','Facilities Services','Farming','Financial Services','Fine Art','Fishery','Food Production','Food/Beverages','Fundraising','Furniture',
'Gambling/Casinos','Glass/Ceramics/Concrete','Government Administration','Government Relations','Graphic Design/Web Design','Health/Fitness','Higher Education/Acadamia',
'Hospital/Health Care','Hospitality','Human Resources/HR','Import/Export','Individual/Family Services','Industrial Automation','Information Services','Information Technology/IT',
'Insurance','International Affairs','International Trade/Development','Internet','Investment Banking/Venture','Investment Management/Hedge Fund/Private Equity','Judiciary',
'Law Enforcement','Law Practice/Law Firms','Legal Services','Legislative Office','Leisure/Travel','Library','Logistics/Procurement','Luxury Goods/Jewelry','Machinery','Management Consulting',
'Maritime','Market Research','Marketing/Advertising/Sales','Mechanical or Industrial Engineering','Media Production','Medical Equipment','Medical Practice','Mental Health Care',
'Military Industry','Mining/Metals','Motion Pictures/Film','Museums/Institutions','Music','Nanotechnology','Newspapers/Journalism','Non-Profit/Volunteering','Oil/Energy/Solar/Greentech',
'Online Publishing','Other Industry','Outsourcing/Offshoring','Package/Freight Delivery','Packaging/Containers','Paper/Forest Products','Performing Arts','Pharmaceuticals','Philanthropy',
'Photography','Plastics','Political Organization','Primary/Secondary Education','Printing','Professional Training','Program Development','Public Relations/PR','Public Safety','Publishing Industry',
'Railroad Manufacture','Ranching','Real Estate/Mortgage','Recreational Facilities/Services','Religious Institutions','Renewables/Environment','Research Industry','Restaurants',
'Retail Industry','Security/Investigations','Semiconductors','Shipbuilding','Sporting Goods','Sports','Staffing/Recruiting','Supermarkets','Telecommunications','Textiles',
'Think Tanks','Tobacco','Translation/Localization','Transportation','Utilities','Venture Capital/VC','Veterinary','Warehousing','Wholesale','Wine/Spirits','Wireless','Writing/Editing']
	

def userdetails():
    # column 1 
    user_id = random.choices(useriddata, weights=None, cum_weights=None, k=1000)
    # column 4 
    language_Proffesional = random.choices(languageProffesional, weights=None, cum_weights=None, k=1000)
    # column 5 
    language_option1 = random.choices(languagedata, weights=None, cum_weights=None, k=1000)
    # column 6 
    language_option2 = random.choices(languagedata, weights=None, cum_weights=None, k=1000)
    
    user_profile = {'user_id': user_id , 'language_Profesional' : language_Proffesional, 'language_option1' : language_option1, 'language_option2' : language_option2}
    
    return user_profile 


def actionGen():
    user = random.choices(userdetails()['user_id'], weights=None, cum_weights=None, k=30000)
    # column 2 random.choices(sequence, weights=None, cum_weights=None, k=1)
    timestamp = random.choices(timedata, weights=None, cum_weights=None, k=30000)
    # column 3 
    action = random.choices(actiondata, weights=None, cum_weights=None, k=30000)
    # column 7
    interests = random.choices(sample(interestsdata, 7), weights=None, cum_weights=None, k=30000)
    # column 14 
    domains = random.choices(sample(domainsdata, 3), weights=None, cum_weights=None, k=30000)
    
    user_dataset = { 'timestamp' : timestamp, 'action' : action, 'user_id': user ,'interests' : interests , 'domains' : domains}
    
    return user_dataset

userProfiledf = pd.DataFrame(userdetails()) 

userDatasetdf = pd.DataFrame(actionGen())


#merged_dataset = userDatasetdf.combine_first(userProfiledf)
#merged_dataset = pd.concat([userDatasetdf, userProfiledf])
#merged_dataset.to_csv('actionSimulationDataset_merged.csv')

userProfiledf.to_csv('actionSimulationDataset_profile.csv')
userDatasetdf.to_csv('actionSimulationDataset_action.csv')
