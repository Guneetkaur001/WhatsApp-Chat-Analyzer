from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # 1. Number of messages
    num_messages = df.shape[0]

    # 2. Total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())
    
    # 3. Number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>'].shape[0]

    # 4. Number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    df = df[df['user'] != 'group_notification']  
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100, 2).reset_index().rename(
        columns={'index':'name', 'user':'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
        
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time 
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    period_order = [
        '00-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10',
        '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17-18',
        '18-19', '19-20', '20-21', '21-22', '22-23', '23-00'
    ]

    df['period'] = pd.Categorical(df['period'], categories=period_order, ordered=True)
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message',
                                aggfunc='count').fillna(0)
    return user_heatmap

def sentiment(d):
    if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
        return 1
    if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
        return -1
    if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
        return 0

def analyze_sentiment(df):
    """Calculate sentiment scores for all messages"""
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['message'].apply(lambda x: sia.polarity_scores(str(x)))
    df['positive'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    df['negative'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['neutral'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    return df

def classify_sentiment(compound_score):
    """Classify sentiment based on compound score"""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_summary(selected_user, df):
    """Get sentiment distribution for selected user"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = analyze_sentiment(df)
    df['sentiment'] = df['compound'].apply(classify_sentiment)
    
    sentiment_counts = df['sentiment'].value_counts()
    return {
        'Positive': sentiment_counts.get('Positive', 0),
        'Negative': sentiment_counts.get('Negative', 0),
        'Neutral': sentiment_counts.get('Neutral', 0)
    }

def get_sentiment_timeline(selected_user, df):
    """Get daily sentiment timeline"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = analyze_sentiment(df)
    df['sentiment'] = df['compound'].apply(classify_sentiment)
    
    timeline = df.groupby(['only_date', 'sentiment']).size().unstack().fillna(0)
    return timeline
    

def get_sentiment_wordcloud(selected_user, df, sentiment_type):
    """Generate wordcloud for specific sentiment"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    df = analyze_sentiment(df)
    df['sentiment'] = df['compound'].apply(classify_sentiment)
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']
    temp = temp[temp['sentiment'] == sentiment_type]
    
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    
    temp['message'] = temp['message'].apply(remove_stop_words)
    text = ' '.join(temp['message'].astype(str))
    
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(text)

# Sentiment-specific analysis functions
def week_activity_map_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['day_name'].value_counts()

def month_activity_map_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['month'].value_counts()

def activity_heatmap_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def daily_timeline_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def monthly_timeline_sentiment(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def percentage(df, k):
    df = round((df['user'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df

def most_common_words_sentiment(selected_user, df, k):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df