import re
import pandas as pd
import numpy as np

##___________________________________________DATASETS___________________________________________
#   Specific methods for read/elaborate the challenge's datasets

def create_text_user(data):

    user_conversations = []
    
    for conversation in data:
        all_conversation = re.findall(r'"(.*?)"', conversation)
        user_conversation = all_conversation[3] + '. ' + all_conversation[7]
        user_conversations.append(user_conversation)

    return user_conversations


#___________________________________________HASHTAHGS___________________________________________
#   For each hashtag present in the training set, a score has been computed in order to identify
#   it as negative or positive, according to the associated hard label.
#   The overall score associated to a sample is obtained through the mean of the scores of the
#   contained hastags.


def get_hastags(df, column = 'Text'):
    """ Method that isolate the hashatg present within a text.

    Args:
        df (dataframe): dataframe with the column in which to search for hashtags
        column (string): column in which to search for hashtags
    Returns:
        df (dataframe): same dataframe with an additional colum containing the list of hashtags (list of strings)
    """
    df['Hashtags'] = df[column].apply(lambda x: re.findall(r"#(\w+)", x)) #add a column with list of hashtags
    #df = pd.concat([df, df.Hashtags.str.join(',').str.get_dummies(sep=',').astype(bool)], axis=1) # add a boolean column for each hashtag
    return df

def compute_hashtags_scores(df):
    """
    Compute a score for each hashtag in the dataframe. 
    The score of an hashtag H is computed as the summatory, for each tweet, of the subscores(H) multiplied by the number
    of the occurences of an hashtag H in T, divided by the total number of occurrences of the hashtags (in the whole 
    dataset). That correspond to the weighted average of the scores.
    The subscore(H) is the difference between the number of annotators that labbeled positive and the number of 
    annotators that labbeled negative an hashtag H, divided by the number of Annotatators of Tweet T.
    The computes score is bounded between [-1; 1]

    Args:
        df (dataframe): dataframe containign the text to process

    Returns:
        dataframe: return a dataframe with the following columns:
                        Hashtag: Hashtag name
                        Number: Number of occurrencies of the hashtag in the training data
                        Score: final score
    """
    hashtags_list = [item for sublist in list(df.Hashtags) for item in sublist]
    #pos_hashtags = [item for sublist in list(df.loc[df['Hard_label']=='0',:].Hashtags) for item in sublist]
    #neg_hashtags = [item for sublist in list(df.loc[df['Hard_label']=='1',:].Hashtags) for item in sublist]
    hash_df = pd.DataFrame(set(hashtags_list), columns=['Hashtag'])
    hash_df['Number'] = hash_df['Hashtag'].apply(lambda x: hashtags_list.count(x))
    hash_df['Score'] = 0

    for _, row in df.iterrows(): #for each sample
        for el in set(row.Hashtags): #for each hashtag in the sample
            partial_score = (row.Annotations.count('1')-row.Annotations.count('0'))/int(row.NumAnnotations)
            hash_df.loc[hash_df['Hashtag']==el, 'Score'] = hash_df.loc[hash_df['Hashtag']==el, 'Score'] + (partial_score * row.Hashtags.count(el)) # partial_score multiplyed by the occurrences in the sample

    hash_df['Score'] = hash_df.Score/hash_df.Number
    return hash_df
   
def get_hashtag_score_for_sample(df):
  #compute a score for each sample, as a mean of the score of the contained hashtags (0 if no hashtag is present)
  hash_df = compute_hashtags_scores(df) 
  df['hashtag_score']=0
  for index, row in df.iterrows():
    tmp = 0
    if row.Hashtags:
      for el in row.Hashtags:
        #check if the hashtag is present in the training dataset
        if len(hash_df.loc[hash_df['Hashtag']==el, 'Score'].values):
          tmp = tmp + hash_df.loc[hash_df['Hashtag']==el, 'Score'].values[0]
      df.loc[index, 'hashtag_score']=tmp/len(row.Hashtags)
  return df

#___________________________________________EMOTIONS___________________________________________
#   Emotions are computer with the NRC Emotion Lexicon according to 8 different emotions:
#    anger, fear, expectation, trust, surprise, sadness, joy and disgust

def count_words_per_emotions(lang, words, nrc_df):
    """compute the emotion vector for the sentence given as input.

    Args:
        lang (String): language of the text, in the form ex.'English (en)'
        words (list): list of words to elaborate
        nrc_df (dataframe): dataframe with the NRC Emotion Lexicon 

    Returns:
        array: 8 dimentional array with a score for each emotion
    """
    count = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    for word in words:
        if word.lemma:
            if lang =='Arabic (ar)': #no lower case
                if word.lemma in nrc_df[lang].values:
                    # prendo la riga corrispondente a quella parola e sommo vettore emozioni
                    count += np.array(nrc_df.loc[nrc_df[lang] == word.lemma].values.tolist()[0][-8:])
            else:
                if word.lemma.lower() in nrc_df[lang].values:
                    # prendo la riga corrispondente a quella parola e sommo vettore emozioni
                    count += np.array(nrc_df.loc[nrc_df[lang] == word.lemma.lower()].values.tolist()[0][-8:])
    return count.tolist()

def get_emotions_for_sample(lang, text, nrc_df, nlp):
    """ Method to elaborate the data as provided from the challenge organizer in order to execute the
    count_words_per_emotions method.
    NB:
        Call:
            df['emotions'] = df.progress_apply(lambda x: get_emotions_for_sample(x.Lang, x.Text, emotion_df), axis=1)
        To show progresses:
            from tqdm import tqdm
            tqdm.pandas()
        
    Args:
        lang (language): language of the text, in the abbreviated form ex. en for English
        text (string): text to be processed 
        nrc_df (dataframe): dataframe with the NRC Emotion Lexicon 
        nlp: stanza pipeline
            can be obtained throug:
                import stanza
                nlp = stanza.Pipeline('en', processors={"pos": "default", 'ner': 'conll03',"tokenize": "spacy",  }, use_gpu=True) 


    Returns:
        array: 8 dimentional array with a score for each emotion
    """
  
    doc = nlp(text)
    words = flatten_list(doc.sentences)

    # Lista delle parole appiattita
    if lang == 'en':
        lang = 'English (en)'
    if lang == 'ar':
        lang = 'Arabic (ar)'
    return count_words_per_emotions(lang, words, nrc_df)

#___________________________________________AGREEMENT VALUES___________________________________________
def mapping_percentage(perc, num_annotators):
    """ return the closest (the absolute value of the difference is the minimum) value between the
    feasible ones acccording to the number of annotators.

    Args:
        perc (float): percentage of agreement predicted by the model
        num_annotators (int): number of annotators for a given task

    Returns:
        float: corrected percentage of agreement
    """
    feasible_values = []
    for x in range(num_annotators+1): #from 0 to num_annotators
        feasible_values.append(x/num_annotators)

    return round(min(feasible_values, key=lambda x:abs(x-perc)), 2)
    #TODO: check: dovrebbe arrotondare per eccesso se superiore a 5



#___________________________________________GENERAL___________________________________________
def flatten_list(sentences):
    return [word for sent in sentences for word in sent.words]