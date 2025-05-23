import re
import pandas as pd

def preprocess(data):
    # Handle both 12-hour format with AM/PM (your format)
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m\s-\s'
    
    messages = re.split(pattern, data)[1:]  #Splits raw chat data into messages using the regex pattern.# Skip first empty element(text before first timestamp)[1:]
    dates = re.findall(pattern, data)       #Extracts date-time strings from raw data.

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    # Clean date strings (keep your 12-hour format handling)
    df['message_date'] = df['message_date'].str.replace(' - ', '', regex=False)
    df['message_date'] = df['message_date'].str.replace('\u202f', ' ', regex=False)
    
    # Convert to datetime with 12-hour format; Handles day-first dates
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p')
    except:
        # Fallback to other common formats if needed ; Handles month-first dates
        try:
            df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p')
        except:
            # Final fallback - let pandas try to infer (Automatic Inference)
            df['message_date'] = pd.to_datetime(df['message_date'])

    df.rename(columns={'message_date': 'date'}, inplace=True)
#seperate users and messages
    users = []    #list
    messages = []

    if 'user_message' in df.columns:  
        for message in df['user_message']:
            # Improved message splitting (from both versions)
            #Separating Users from Messages; split(':') would break messages containing colons (e.g., "https://example.com").
            #Regex: Splits only on the first colon after a username
            entry = re.split(r'([^:]+):\s', message, maxsplit=1)
            if len(entry) > 2:
                users.append(entry[1].strip())
                messages.append(entry[2].strip())
            else:
                users.append("group_notification")
                messages.append(entry[0].strip())
    
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Date features extraction (keeping your existing columns)
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create time periods (keeping your 12-hour based logic)
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period

    return df