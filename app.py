import nltk
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon at the start
nltk.download('vader_lexicon')

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert to string
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    
    # Creating different columns for (Positive/Negative/Neutral)
    sentiments = SentimentIntensityAnalyzer()
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]  # Neutral
    df["compound"] = [sentiments.polarity_scores(i)["compound"] for i in df["message"]]
    
    # Apply sentiment function
    df['value'] = df.apply(lambda row: helper.sentiment(row), axis=1)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    
    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.title("Total Chats")
            st.title(num_messages)
        with col2:
            st.title("Total Words")
            st.title(words)
        with col3:
            st.title("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.title("Links Shared")
            st.title(num_links)


        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        
        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")    
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()  
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Finding Busiest Users in Group (Group Level)
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
           
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)
       
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        # ============= SENTIMENT ANALYSIS SECTION =============
        st.markdown("---")
        st.title("Sentiment Analysis")
        
        # Get sentiment summary
        sentiment_summary = helper.get_sentiment_summary(selected_user, df)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive Messages", sentiment_summary['Positive'])
        with col2:
            st.metric("Neutral Messages", sentiment_summary['Neutral'])
        with col3:
            st.metric("Negative Messages", sentiment_summary['Negative'])
        
        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
            busy_month = helper.month_activity_map_sentiment(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
            busy_month = helper.month_activity_map_sentiment(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
            busy_month = helper.month_activity_map_sentiment(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
            busy_day = helper.week_activity_map_sentiment(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
            busy_day = helper.week_activity_map_sentiment(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
            busy_day = helper.week_activity_map_sentiment(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                user_heatmap = helper.activity_heatmap_sentiment(selected_user, df, 1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")

        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                user_heatmap = helper.activity_heatmap_sentiment(selected_user, df, 0)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                user_heatmap = helper.activity_heatmap_sentiment(selected_user, df, -1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)
            daily_timeline = helper.daily_timeline_sentiment(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)
            daily_timeline = helper.daily_timeline_sentiment(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)
            daily_timeline = helper.daily_timeline_sentiment(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)
            timeline = helper.monthly_timeline_sentiment(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)
            timeline = helper.monthly_timeline_sentiment(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)
            timeline = helper.monthly_timeline_sentiment(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Percentage contributed
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                x = helper.percentage(df, 1)
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                y = helper.percentage(df, 0)
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                z = helper.percentage(df, -1)
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # Sentiment Pie Chart
        fig, ax = plt.subplots()
        ax.pie(
            [sentiment_summary['Positive'], sentiment_summary['Neutral'], sentiment_summary['Negative']],
            labels=['Positive', 'Neutral', 'Negative'],
            autopct="%1.1f%%",
            colors=['#4CAF50', '#FFC107', '#F44336']
        )
        st.pyplot(fig)
        
        # Sentiment Timeline
        st.subheader("Sentiment Over Time")
        sentiment_timeline = helper.get_sentiment_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 5))
        sentiment_timeline.plot(ax=ax)
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
        st.pyplot(fig)
        
    
        # Sentiment Word Clouds
        st.subheader("Sentiment Word Clouds")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Positive Words**")
            pos_wc = helper.get_sentiment_wordcloud(selected_user, df, 'Positive')
            fig, ax = plt.subplots()
            ax.imshow(pos_wc)
            ax.axis('off')
            st.pyplot(fig)
        with col2:
            st.markdown("**Neutral Words**")
            neu_wc = helper.get_sentiment_wordcloud(selected_user, df, 'Neutral')
            fig, ax = plt.subplots()
            ax.imshow(neu_wc)
            ax.axis('off')
            st.pyplot(fig)
        with col3:
            st.markdown("**Negative Words**")
            try:
                neg_wc = helper.get_sentiment_wordcloud(selected_user, df, 'Negative')
                fig, ax = plt.subplots()
                ax.imshow(neg_wc)
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")

        # Most Common Words by Sentiment
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                most_common_df = helper.most_common_words_sentiment(selected_user, df, 1)
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")
        with col2:
            try:
                most_common_df = helper.most_common_words_sentiment(selected_user, df, 0)
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")
        with col3:
            try:
                most_common_df = helper.most_common_words_sentiment(selected_user, df, -1)
                st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
                st.info("This usually happens when there's not enough data for this specific user/sentiment combination")