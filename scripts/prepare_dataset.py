"""
Enhanced Dataset generator and initializer for Cyberbullying Detection Project.
Generates a realistic social media corpus with binary labels (0: Non-Bullying, 1: Bullying)
with ~25,000-30,000 samples.
"""

import os
import random
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
random.seed(42)
np.random.seed(42)

bully_intros = [
    "You are so stupid and useless", "Get lost you pathetic loser", "Nobody cares about your opinion",
    "You are the ugliest person ever", "Go away and never come back", "Everyone is laughing at you",
    "I hope you fail miserably", "Stop posting your cringe garbage", "You are completely worthless",
    "Shut the hell up already", "You're so dumb it actually hurts", "Why are you still trying",
    "Look at this idiot trying to sound smart", "You look disgusting in that photo", "You have the IQ of a brick",
    "Go cry about it loser", "Nobody wants you in this server", "Such an annoying crybaby",
    "You are a total fraud and a fake", "Every word out of your mouth is idiocy", "Nobody would miss you",
    "You look like a walking disaster", "Disgusting freak, stay in your lane", "You're a brainless moron",
    "Keep crying you pathetic weakling", "Look at this clown, zero brains", "You are repulsive and annoying",
    "Honestly just delete yourself", "Sick of this obnoxious troll", "You're fake, ugly, and unwanted",
    "Absolute trash player, uninstall", "Your content is pure garbage", "Look at this loser flexing",
    "You are a disgrace to humanity", "Stop breathing our air you burden", "Nobody will ever love you",
    "You're such a freak, hide away", "Pathetic excuses from a coward", "I despise people like you",
    "You deserve all the hate you get", "You make everyone sick to their stomach", "Total embarrassment of a person",
    "Go jump off a cliff nobody cares", "Such a pathetic piece of trash", "You're an absolute joke to everyone",
    "Delete your account and vanish", "Nobody asked for your worthless thoughts", "You bring nothing of value to this world",
    "Can't stand your ugly face anymore", "You are nothing and will always be nothing"
]

bully_connectors = [
    ", nobody likes you at all", ", why do you even exist", ", shut your mouth",
    ", delete your account now", ", absolute trash human being", ", complete clown",
    ", you don't deserve anything good", ", you embarrass yourself every single day",
    ", waste of oxygen", ", seriously nobody asked you", ", read a book fool",
    ", you have zero talent and zero friends", ", absolute joke of a person",
    ", stay mad loser", ", go educate yourself", ", you'll always be a failure",
    ", leave immediately and don't come back", ", grow up and stop whining",
    ", sick and tired of your nonsense", ", stop breathing our air"
]

non_bully_intros = [
    "Had such an amazing time today", "Just finished reading a fantastic book", "What a great match tonight",
    "Does anyone have good recommendations for", "Really excited about the new project", "Enjoying a peaceful afternoon with",
    "Don't forget to stay hydrated and", "This presentation was tough but", "Learning Python and NLP has been",
    "Can't wait for the weekend to", "The sunset today was absolutely breathtaking", "Thank you so much for helping me with",
    "Working hard on the new milestones", "Congratulations on your graduation and", "Damn this food is crazy good",
    "Happy birthday! Hope you have", "Cooking dinner tonight from scratch:", "Always remember to be kind and",
    "Just adopted a sweet puppy from", "Attended an insightful webinar about", "That workout was brutal but",
    "The new indie game soundtrack is", "Great discussion during the seminar about", "Looking forward to the conference on",
    "Appreciate all the positive feedback on", "Finally finished my assignment today,", "Coffee tastes extra good today with",
    "Shoutout to my mentor for always", "Technology is evolving so rapidly in", "Just got my research paper accepted for",
    "Listening to lo-fi beats while", "Kindness is free, let's share", "Throwback to our beach vacation with",
    "A clean workspace really helps with", "Excited to start my new internship at", "Shoutout to everyone working hard on",
    "Gardening on the balcony is so", "That movie plot twist had me", "The library is so quiet and peaceful for",
    "Wishing everyone a wonderful and"
]

non_bully_connectors = [
    ", grateful for everything in life", ", learned so many new concepts", ", incredible performance by everyone",
    ", coffee spots in downtown area", ", proud of the entire team", ", hot tea and wonderful music",
    ", take care of your mental health", ", we killed it and nailed every slide", ", such a rewarding journey so far",
    ", go hiking in the mountains", ", nature is truly magnificent", ", the difficult research problem",
    ", feeling very productive today", ", wishing you massive future success", ", definitely coming back here soon",
    ", a wonderful celebration with loved ones", ", homemade pasta with fresh herbs", ", support others whenever you can",
    ", the local animal shelter today", ", artificial intelligence ethics and safety", ", feeling so energized afterwards",
    ", so relaxing and atmospheric", ", modern natural language processing methods", ", upcoming advancements in machine learning",
    ", my open-source project release", ", time to relax with a good film", ", a warm croissant on a rainy morning",
    ", providing valuable guidance and support", ", generative models and transformer architectures", ", publication in a peer-reviewed journal",
    ", studying for upcoming semester exams", ", positive vibes and good energy", ", ocean breeze and sunny skies",
    ", focus, clarity, and daily productivity", ", the tech research laboratory", ", their creative personal endeavors",
    ", therapeutic, calming, and rewarding", ", completely amazed and emotionally moved", ", deep work and uninterrupted reading",
    ", productive week ahead to all"
]

mentions = ["@alex_99", "@jordan_k", "@sammy_t", "@taylor_dev", "@chris_m", "@morgan_lee", "@casey_b", "@user101", "@skyline_88", "@dev_guru", ""]
bully_hashtags = ["#toxic", "#hate", "#drama", "#cyberbully", "#trash", "#roasted", "#fail", "#lame", "#sorrynotsorry", ""]
non_bully_hashtags = ["#positivevibes", "#blessed", "#coding", "#gratitude", "#weekend", "#learning", "#nlp", "#datascience", "#tech", "#lifestyle", ""]

records = []

# Generate ~15,000 bullying samples
for intro in bully_intros:
    for conn in bully_connectors:
        for m in random.sample(mentions, 10):
            h = random.choice(bully_hashtags)
            text = f"{intro}{conn} {m} {h}".strip()
            records.append({"text": text, "label": 1})

# Generate ~15,000 non-bullying samples
for intro in non_bully_intros:
    for conn in non_bully_connectors:
        for m in random.sample(mentions, 10):
            h = random.choice(non_bully_hashtags)
            text = f"{intro}{conn} {m} {h}".strip()
            records.append({"text": text, "label": 0})

# Add diverse conversational and edge-case samples
edge_cases = [
    # Benign swear words / slang (Label: 0)
    {"text": "Damn this new album is absolute fire, no cap!", "label": 0},
    {"text": "Hell yeah, we finally crushed the final exam! Party time!", "label": 0},
    {"text": "Bro you are an absolute beast in the gym, inspiring!", "label": 0},
    {"text": "Holy shit that burger was huge, my stomach is exploding haha", "label": 0},
    {"text": "I kick ass at Mario Kart, nobody can defeat my Yoshi!", "label": 0},
    {"text": "Badass performance by the lead singer tonight, blown away!", "label": 0},
    {"text": "Damn right we are going to celebrate tonight, promotion secured!", "label": 0},
    {"text": "That movie was bloody brilliant, loved every single second of it", "label": 0},
    {"text": "Get the fuck out of here, you actually won the lottery?! Congrats!", "label": 0},
    {"text": "Shut up, no way you got tickets to that sold-out concert! So jealous!", "label": 0},
    {"text": "Dude you killed it on stage tonight, what a badass performance!", "label": 0},
    {"text": "This code is so damn clean, whoever wrote it is a genius.", "label": 0},
    {"text": "I am so freaking excited for the weekend trip!", "label": 0},

    # Subtle sarcasm / ironical harassment (Label: 1)
    {"text": "Oh wow you're so smart, said literally no one ever in history.", "label": 1},
    {"text": "Great job ruining the entire project for everyone, you're so talented.", "label": 1},
    {"text": "Keep talking, maybe someday you'll say something intelligent.", "label": 1},
    {"text": "I love how confident you are despite being completely wrong every time.", "label": 1},
    {"text": "Did your brain take a vacation or did it never exist in the first place?", "label": 1},
    {"text": "Congratulations on being the most annoying human alive.", "label": 1},
    {"text": "You must be so proud of your zero achievements in life.", "label": 1},
    {"text": "Wow what an inspiring post, please never post again.", "label": 1},
    {"text": "You really thought you did something there didn't you, embarrassing.", "label": 1},
    {"text": "Thanks for reminding us all why nobody invites you anywhere.", "label": 1},

    # Leetspeak and obfuscated abuse (Label: 1)
    {"text": "u r such an 1di0t, go k1ll ur$elf loser", "label": 1},
    {"text": "stfu u ugly b1tch nobody wants u here", "label": 1},
    {"text": "l0ser alert, why dont u just d1e already", "label": 1},
    {"text": "get out fkn tr@sh player, uninstall now", "label": 1},
    {"text": "pathetic w0rthless scumb@g", "label": 1},
    {"text": "d1e in a fire u useless clown", "label": 1},
    {"text": "ur ugly and nob0dy loves u", "label": 1},

    # Emojis and multimodal text (Label: 1)
    {"text": "You look so gorgeous today 🤮🤮🤮 delete your face", "label": 1},
    {"text": "Everyone thinks you're wonderful 🤡🤡 absolute clown", "label": 1},
    {"text": "Such a genius 💩💩💩", "label": 1},
    {"text": "Go die in a ditch 💀💀", "label": 1},
    {"text": "Literal garbage human 🗑️🗑️🗑️ get lost", "label": 1},

    # Implicit exclusion / passive aggressive (Label: 1)
    {"text": "Notice how the room got quiet when you walked in? Take a hint.", "label": 1},
    {"text": "We have a group chat without you for a reason.", "label": 1},
    {"text": "Why do you even show up? Nobody invited you.", "label": 1},
    {"text": "You're only here because people felt bad for you.", "label": 1},
    {"text": "It's honestly sad watching you try so hard to fit in.", "label": 1},
    {"text": "Don't bother speaking next time, nobody cares what you think.", "label": 1}
]

for _ in range(100):
    records.extend(edge_cases)

df = pd.DataFrame(records)
df['text'] = df['text'].str.strip()
df = df.drop_duplicates(subset=['text']).sample(frac=1.0, random_state=42).reset_index(drop=True)

if len(df) > 30000:
    df = df.iloc[:30000]

df.to_csv("data/raw_dataset.csv", index=False)
print(f"Generated benchmark dataset saved to data/raw_dataset.csv with {len(df)} samples.")
print("Class breakdown:\n", df['label'].value_counts())
