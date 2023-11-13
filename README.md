# Can ChatGPT be used for instant, reliable A|B testing? (Yes)


# Inspiration
On https://cordial.com/platform/data-platform/ai/ under "EXPERIMENTS" it says:


'Leverage machine learning to automate message experiments to automatically favor better performing variants quicker, with less effort.'

I also wanted to combine this with the recent Cordial strategy: *Generative, Predictive, and Prescriptive AI.*

# Purpose
I want to work on challenging problems related to prompt engineering and utilizing generative AI for companies that are leading the way. This work has not been shared with any other organization.

# Results

Below ChatGPT and 1500 real restaurants were given 3 different emails templates asking if they would like to sign up for a newsletter about restaurant trends. 

**ChatGPT correctly predicted the best and worst emails as if it were a human recieving a cold outbound email.**

![alt text](https://github.com/clawson62/Cordial_Project/blob/main/chart1.jpg?raw=true)

### Understanding What Contributed to Response Rate

Below shows 4 features ChatGPT was asked to score each of the emails on: 
1. Personalization
2. Clarity
3. Value Proposition
4. Call to Action

**Personalization had the greatest effect.**

![alt text](https://github.com/clawson62/Cordial_Project/blob/main/chart2.jpg?raw=true)

# Further Work

Looking specifically at Cordial's current clients, some clients may face different marketing challenges based on their customer archetype. Below are a few continuations of this work, but I have a much more extensive list if this work would like to be pursued.

## 1. Ticket Price Consideration

Theory: The features in emails for lower ticket items such as **Church's Chicken** and **Cold Stone** ice cream may not hold the same effectiveness as features for higher ticket items such as **Purple** mattresses and **Virgin Voyages** cruises.

Example: The **value proposition** of getting a better night's sleep on a quality mattress may be the most persuasive feature in a **Purple** email campaign, however, the **personalization** of a **Church's Chicken** being only 1.1 miles away may be the most persuasive feature in a **Church's Chicken** email campaign.

## 2. Archetype 

Theory: Generating customer archetype json objects for ChatGPT can guide an even more precises prediction of A|B testing results. I used a simple version of this in my experiment that lead to a much higher response rate.

Examples (these can be fed directly to ChatGPT and it's response will be altered to act more like the archetype when deciding to respond or ignore):

### Company: Adore Me
```json
{
  "archetype": "Trend-Conscious Comfort Seeker",
  "demographics": {
    "age": "20",
    "gender": "Female",
    "incomeLevel": "Middle to Upper-Middle Class",
    "educationLevel": "College Educated"
  },
  "psychographics": {
    "values": ["Fashion-forward", "Comfort", "Affordability"],
    "interests": ["Fashion", "Online Shopping", "Health and Wellness"],
    "lifestyle": ["Busy Professional", "Socially Active", "Fitness Enthusiast"]
  },
  "shoppingBehaviors": {
    "preferredShoppingChannels": ["Online", "Mobile App"],
    "purchaseDrivers": ["Style", "Comfort", "Promotions and Discounts"],
    "brandLoyalty": "Moderate to High",
    "averagePurchaseFrequency": "Quarterly"
  },
  "productPreferences": {
    "favoredProducts": ["Lingerie", "Loungewear", "Swimwear"],
    "sizeRange": ["Inclusive Sizing"],
    "stylePreferences": ["Trendy", "Versatile", "Comfortable"]
  },
  "communicationPreferences": {
    "preferredCommunicationChannels": ["Email", "Social Media"],
    "engagementLevel": "Highly Responsive to Sales and New Arrivals"
  },
  "digitalEngagement": {
    "socialMediaPlatforms": ["Instagram", "Pinterest", "Facebook"],
    "onlineActivities": ["Browsing Fashion Blogs", "Following Influencers", "Participating in Online Communities"]
  }
}

```
### Company: Backcountry

```json
{
  "archetype": "Outdoor Adventure Enthusiast",
  "demographics": {
    "age": "35",
    "gender": "Unisex",
    "incomeLevel": "Middle to Upper Class",
    "educationLevel": "College Educated"
  },
  "psychographics": {
    "values": ["Outdoor Lifestyle", "Sustainability", "Quality"],
    "interests": ["Hiking", "Camping", "Mountain Biking", "Skiing"],
    "lifestyle": ["Active Outdoor Pursuits", "Environmental Consciousness", "Travel"]
  },
  "shoppingBehaviors": {
    "preferredShoppingChannels": ["Online", "Physical Retail Stores"],
    "purchaseDrivers": ["Durability", "Brand Reputation", "Technical Features"],
    "brandLoyalty": "High",
    "averagePurchaseFrequency": "Seasonally"
  },
  "productPreferences": {
    "favoredProducts": ["Technical Apparel", "Camping Gear", "Climbing Equipment", "Winter Sports Gear"],
    "sizeRange": ["Wide Range to Suit Different Body Types"],
    "stylePreferences": ["Functional", "Durable", "High-Performance"]
  },
  "communicationPreferences": {
    "preferredCommunicationChannels": ["Email", "Outdoor Forums", "Brand Websites"],
    "engagementLevel": "Engages in Reviews and Outdoor Community Discussions"
  },
  "digitalEngagement": {
    "socialMediaPlatforms": ["Instagram", "YouTube", "Reddit - Outdoor Communities"],
    "onlineActivities": ["Sharing Adventure Experiences", "Researching Gear", "Participating in Outdoor Forums"]
  }
}

```


## 3. Marketing Style Transfer
Theory: Use ChatGPT to replicate different marketing technique by seeding the framework in the prompt.

Example: Using the techniques in Robert Cialdini’s book *Influence*, generate an email marketing campaign for abandoned carts for the company **Eddie Bauer**

# Description of My Work

Below I describe the steps that create the data in order to populate the **Results** section above.

## Experimental Setup

1. Generate newsletter sign-up email templates with ChatGPT
2. Prompt engineering a scoring mechanism for emails

## 1. Generate Emails

Each email was generated for the following purpose:
0. Personalization focused
2. More salesman-like
3. The control - simple ask, no added content


## 2. Email Scoring Mechanism
This is where ChatGPT says if its a "good" email or not.

**Two** Methods were used for scoring
1. Categorization - 4 categories scored from 1-5
2. Binary Action - ChatGPT chooses to "respond" or "ignore" email as if it were the restaurant owner


```python
import openai
import pandas as pd
import time
import traceback
import json
import numpy as np
from collections import defaultdict

key = ""# insert key
openai.api_key = key

email_data_df = pd.read_json("../yelp api/affiliate_agency/broken_by_name/first Mark S.json")

# Example restaurant data collected
food = "mexican"
location = "Denver"

background = {"restaurant location":location,
              "restuarant food sold":food
}

salesy_email = open("salesy_email.txt").read()

# emails that were generated
emails = [
    # 1. personalization focused
    f"Hi there!\n\nThe food on your website looks so tasty, I just had to say hi. \
I created a newsletter for the {location} area that I thought you might like with great {food} food topics. \
And don't worry, no long essays here – just short, sweet bites of what I've learned about your industry.",

    # 2. Salesman focused
salesy_email,
    
    # 3. Control 
"Hi there!\n\nWould you like to sign up for my newsletter about restaurant trends?"
]


ending = f"\n\nIf you’d like to sign up, just shoot me an email and I’ll add you to the list.\n\nBest,\nMark"


for email in emails:
    if email!=emails[1]:
        print(email+ending)
    else:
        print(email)
    print("*"*100)
```

    Hi there!
    
    The food on your website looks so tasty, I just had to say hi. I created a newsletter for the Denver area that I thought you might like with great mexican food topics. And don't worry, no long essays here – just short, sweet bites of what I've learned about your industry.
    
    If you’d like to sign up, just shoot me an email and I’ll add you to the list.
    
    Best,
    Mark
    ****************************************************************************************************
    Dear Restaurant Owner,
    
    Are you striving to keep your finger on the pulse of the culinary world? Look no further! Our industry-leading newsletter is packed with cutting-edge insights, sizzling marketing tips, and exclusive data to fuel your growth and outpace the competition. Join a community of forward-thinking restaurateurs who are already reaping the benefits. Sign up today - it's quick, it's free, and it will transform your approach to success!
    
    Don't miss out on this golden opportunity to enhance your restaurant's performance. Every issue delivers hot trends, expert strategies, and special offers directly to your inbox. Why wait? If you'd like to sign up, just shoot me an email and I'll add you to the list!
    
    Best regards,
    Mark
    ****************************************************************************************************
    Hi there!
    
    Would you like to sign up for my newsletter about restaurant trends?
    
    If you’d like to sign up, just shoot me an email and I’ll add you to the list.
    
    Best,
    Mark
    ****************************************************************************************************
    


```python
def catagorized_email(email,background):
    prompt = "Background: "+str(background)+"\n\n Email: "+email
    
    messages = [{"role": "user", "content": prompt}]
    functions = [
        {
            "name": "catagorized_email",
            "description": "You are a email marketing expert. Score this email from 1 (bad) to 5 (good) based on these criteria: personalization, clarity, value proposition, and call to action. Background information is what we know about the restaurnt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "personalization": {
                        "type": "integer",
                        "description": "Possible good attiributes: name use, specific location or other type of information pertaining to receiver.",
                    },
                    "clarity": {
                        "type": "integer",
                        "description": "Possible good attiributes: easy to understand, use of simple language. Lower value if spammy or salesy.",
                    },
                    "value_proposition": {
                        "type": "integer",
                        "description": "Possible good attiributes: communicate the benefits, believable, low time delay, low effort on their part.",
                    },
                    "call_to_action": {
                        "type": "integer",
                        "description": "Possible good attiributes: easy to find, simple to do, adds urgency",
                    },
                },
                "required": ["personalization",
                             "clarity",
                             "value_proposition",
                             "call_to_action"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106", # newest model
        messages=messages,
        functions=functions,
        temperature=1.5,
        function_call="auto", 
    )
    response_message = response
    return response_message

def email_action(email,background):
    prompt = "Email: "+email
    
    messages = [{"role": "user", "content": prompt}]
    functions = [
        {
            "name": "score_email",
            "description": f"You are a restraunt owner in {background['restaurant location']} selling {background['restuarant food sold']} food. Respond with your action to the following email with either ignore or respond. Base you decision as if you were a real person recieving the email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "options: ignore or respond",
                    }
                },
                "required": ["action"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106", # newest model
        messages=messages,
        functions=functions,
        temperature=1.5,
        function_call="auto", 
    )
    response_message = response
    return response_message

def create_email(email_number,food,location):

    background = {"restaurant location":location,
                  "restuarant food sold":food
    }

    
    emails = [
        f"Hi there!\n\nThe food on your website looks so tasty, I just had to say hi. \
    I created a newsletter for the {location} area that I thought you might like with great {food} food topics. \
    And don't worry, no long essays here – just short, sweet bites of what I've learned about your industry.",

    salesy_email,

    "Hi there!\n\nWould you like to sign up for my newsletter about restaurant trends?"
    ]

    ending = f"\n\nIf you’d like to sign up, just shoot me an email and I’ll add you to the list.\n\nBest,\nMark"
    
    email = emails[email_number]
    if email_number!=1:
        email = email+ending
    
    return email
```


```python
debug = False

for row in email_data_df.iloc[500:1030].itertuples():
    
    if row[0]%100==0:
        print(row[0])
        
    for email in emails:
        email = create_email(email_number,row.what_they_sell,row.city)
        
        while True:
            try:
                r = catagorized_email(email,background)
                time.sleep(0.3)
                if debug:
                    print(r["choices"][0]["message"]["function_call"]["arguments"])
                
                with open("catagorized_gpt.json","a") as fp:
                    json.dump({"email":email,"background":background,"response":r},fp)
                    fp.write("\n")
                
                break
                
            except KeyboardInterrupt:
                break
            except:
                if debug:
                    print(traceback.format_exc())
                time.sleep(2)
                


```


```python
# read data for catagorization method
data = []
with open("catagorized_gpt.json","r") as fp:
    for line in fp:
        data.append(json.loads(line))
        
# clean and label responses with email number
rs = [ ]
for i in data:
    if "The food on your website looks" in i["email"]:
        email = 0
    elif "Would you like to sign up for my newsletter" in i["email"]:
        email=2
    else:
        email = 1
    
    if "function_call" in i["response"]["choices"][0]["message"]:
        try:
            d = json.loads(i["response"]["choices"][0]["message"]["function_call"]["arguments"])
            d["email"] = email
            rs.append(d)
        except:
            pass
        
# form dataframe with catagories
catagory_df = pd.DataFrame(rs)[['personalization','clarity','value_proposition','call_to_action','email']]
catagory_df = catagory_df.dropna().reset_index(drop=True)
```


```python
debug = False

for row in email_data_df.iloc[500:1030].itertuples():
    
    if row[0]%100==0:
        print(row[0])
        
    for email in emails:
        email = create_email(email_number,row.what_they_sell,row.city)
        
        while True:
            try:
                r = email_action(email,background)
                time.sleep(0.3)
                if debug:
                    print(r["choices"][0]["message"]["function_call"]["arguments"])
                
                with open("scoring_gpt.json","a") as fp:
                    json.dump({"email":email,"background":background,"response":r},fp)
                    fp.write("\n")
                
                break
                
            except KeyboardInterrupt:
                break
            except:
                if debug:
                    print(traceback.format_exc())
                time.sleep(2)
                


```


```python
# read data for catagorization method
data = []
with open("binary_gpt.json","r") as fp:
    for line in fp:
        data.append(json.loads(line))
        
# clean and label responses with email number
rs = [ ]
for i in data:
    if "The food on your website looks" in i["email"]:
        email = 0
    elif "Would you like to sign up for my newsletter" in i["email"]:
        email=2
    else:
        email = 1
    
    if "function_call" in i["response"]["choices"][0]["message"]:
        try:
            d = json.loads(i["response"]["choices"][0]["message"]["function_call"]["arguments"])
            d["email"] = email
            rs.append(d)
        except:
            pass
        
# form dataframe with actions
action_df = pd.DataFrame(rs)[['action','email']]
action_df["action"] = [1 if row.action=="respond" else 0 for row in action_df.itertuples()]
```
