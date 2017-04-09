import preprocessing
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    #split the data
    #m = preprocessing.Messages('parsed_messages.txt')

    # only for the first time for uncleaned data
    #df = preprocessing.read_messages_dataframe_original('parsed_messages.txt')

    df = preprocessing.read_messages_dataframe('clean_parsed_messages.csv')
    #'message_id', 'province_code', 'city_code', 'gender', 'screen_name', 'original_message', 'word_count', 'words', 'words_tags'
    gender_data = preprocessing.get_specific_columns(df, ['gender', 'words'])

    #printing total number of lines
    #print len(gender_data.index)
    #sample data
    #print gender_data.sample(n=10)


    #printing separate line counts
    #print len(training_data.index), len(development_data.index), len(testing_data.index)

    #gender_labels = training_data.gender.unique()

    #print len(gender_data.index)

    #remove rows with 'nan' as gender and '\N' as words


    #remove later
    #(ran, gender_data) = train_test_split(testing_data, test_size = 0.001)

    gender_data = gender_data.loc[(gender_data['gender'] != 'nan') | (gender_data['words'] != '\N')]

    #print len(clean_gender_data.index)

    df = preprocessing.read_emoticons_dataframe('meta_emoticons.txt')

    #read only emoji column
    emoji_column = preprocessing.get_specific_columns(df, ['emoji'])

    #print emoji_column.sample(n=10)

    emoji_column["emoji"] = emoji_column["emoji"].str.strip(']').str.strip('[')

    emoji_list = emoji_column["emoji"].tolist() 
    gender_data["words"] = gender_data["words"].str.split(' ')

    #gender_data["emojis"] = gender_data["word_list"] - emoji_list

    #gender_data["emojis"] = pd.Series((',').join(emoji_list), gender_data.index)

    #gender_data["emojis"] = gender_data["emojis"].str.split(' ')

    #gender_data["emojis"] = gender_data["emojis"] - gender_data["word_list"]

    emoji_set = set(emoji_list)
    
    #def check_row (val):
    #    return list(set(val).intersection(emoji_set))

    '''
    for index in gender_data.iteritems():
        common = check_row(gender_data.loc[index,'word_list'])
        gender_data.loc[index,'emojis'] = common
    '''

    #gender_data = gender_data[pd.notnull(gender_data['word_list'])]

    gender_data = gender_data.dropna() 

    gender_data['emojis'] = gender_data['words'].apply(lambda x: ','.join(list(set(x).intersection(emoji_set))))

    '''
    for i, row in gender_data.iterrows():
        #print type(row['word_list'])
        #print i, row
        try:
            val = list(set(row['word_list']).intersection(emoji_set))
            gender_data.set_value(i,'emojis', ','.join(val))
        except TypeError:
            print i, row, type(row['word_list'])

    '''

    print gender_data.sample(n=10)
    #print emoji_column.sample(n=10)
    gender_data.to_csv('messages_emojis.csv')

    #Training: 70%, Development: 20%, Testing: 10%
    (training_data, development_data, testing_data) = preprocessing.split_data(gender_data)
    training_data.to_csv('training_messages_emojis.csv')
    development_data.to_csv('development_solution.csv')
    testing_data.to_csv('testing_solution.csv')

    preprocessing.get_specific_columns(development_data, ['gender', 'words']).to_csv('development_set.csv')
    preprocessing.get_specific_columns(testing_data, ['gender', 'words']).to_csv('testing_set.csv')



if __name__ == "__main__":
    main()