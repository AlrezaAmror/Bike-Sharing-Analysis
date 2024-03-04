import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes('bright')
import streamlit as st

day_df = pd.read_csv('data/day.csv')
hour_df = pd.read_csv('data/hour.csv')

season_labels = {
    1 : 'springer',
    2 : 'summer',
    3 : 'fall',
    4 : 'winter'
}

weather_labels = {
    1 : 'Cerah',
    2 : 'Berawan',
    3 : 'HujanRingan',
    4 : 'HujanLebat'
}


# function for denormalization temp and atemp variable
def denorm(y, t_min, t_max):
    x = y * (t_max - t_min) + t_min
    return x

# change dtype of dteday from object to datetime
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# denormalisasi variabel temp
day_df['temp'] = denorm(day_df['temp'], -8, 39)
hour_df['temp'] = denorm(hour_df['temp'], -8, 39)

# denormalisasi variabel atemp
day_df['atemp'] = denorm(day_df['atemp'], -16, 50)
hour_df['atemp'] = denorm(hour_df['atemp'], -16, 50)

# denormalisasi variabel hum
day_df['hum'] = day_df['hum']*100
hour_df['hum'] = hour_df['hum']*100

# denormalisasi variabel windspeed
day_df['windspeed'] = day_df['windspeed']*67
hour_df['windspeed'] = day_df['windspeed']*67

# change format variable to real data
day_df['season'] = day_df['season'].map(season_labels)
hour_df['season'] = hour_df['season'].map(season_labels)
day_df['weathersit'] = day_df['weathersit'].map(weather_labels)
hour_df['weathersit'] = hour_df['weathersit'].map(weather_labels)

# The Dashboard
min_date = day_df['dteday'].dt.date.min()
max_date = day_df['dteday'].dt.date.max()

with st.sidebar:
    st.image('dashboard/logo.png', use_column_width = True, caption = 'Bike Sharing') 
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Interval Waktu',
        min_value= min_date,
        max_value= max_date,
        value=[min_date, max_date]
    )

# create dataframe based the time
main_day_df = day_df[(day_df['dteday'] >= str(start_date)) & (day_df['dteday'] <= str(end_date))]
main_hour_df = hour_df[(hour_df['dteday'] >= str(start_date)) & (hour_df['dteday'] <= str(end_date))]

st.header('Bike Rental Dashboard 🚲')

st.subheader('Daily Rental')
total_rent_perhour = main_hour_df.cnt.sum()/len(main_day_df)
casual_rent_perhour = main_hour_df.casual.sum()/len(main_day_df)
registered_rent_perhour = main_hour_df.registered.sum()/len(main_day_df)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Daily Casual User", value=round(casual_rent_perhour, 2))
with col2:
    st.metric("Average Daily Registered User", value=round(registered_rent_perhour, 2))
with col3:
    st.metric("Average Daily Total User", value=round(total_rent_perhour, 2))

st.subheader('Bike Rental at Certain Hours') #=================================================
plt.figure(figsize = (35,15))
# plt.subplot(2,1,1)
ax = sns.barplot(
    data = main_hour_df, x = 'hr', y = 'cnt', estimator = 'sum', color = 'y',errorbar = None
)
for p in ax.patches:
                    # Valuenya                          # Posisi
    ax.annotate(f'{round(p.get_height(),2)}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3),
                textcoords='offset points')
plt.xlabel(None)
plt.ylabel('None')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
st.pyplot(plt)

# plt.subplot(2,1,2)
plt.figure(figsize = (35,15))
sns.barplot(
    main_hour_df, x="hr", y="registered", estimator="sum", errorbar=None, color = 'g', label = 'registered'
    )
sns.barplot(
    main_hour_df, x="hr", y="casual", estimator="sum", errorbar=None, color = 'y', label = 'casual'
    )
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel(None)
plt.ylabel(None)
plt.legend(fontsize = 20)
st.pyplot(plt)

st.subheader('Average Bike Rental Based On Season') #=================================================
plt.figure(figsize = (15,7))
ax = sns.barplot(
    data = main_day_df, x = 'season', y = 'cnt', estimator = 'mean', errorbar = None, palette = 'inferno', hue = 'weathersit'
)
for p in ax.patches:
                    # Valuenya                          # Posisi
    ax.annotate(f'{round(p.get_height(),2)}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3),
                textcoords='offset points')
plt.xlabel(None)
plt.ylabel('None')
plt.xticks(fontsize = 10)
plt.gca().ticklabel_format(style = 'plain', axis = 'y')
plt.yticks(fontsize = 10)
st.pyplot(plt)

st.subheader('Average Bike Rental Based On Working Day') #=================================================
plt.figure(figsize = (15,7))
ax = sns.barplot(
    data = main_day_df, x = 'workingday', y = 'cnt', estimator = 'mean', errorbar = None, palette = 'inferno', hue = 'weathersit'
)
for p in ax.patches:
                    # Valuenya                          # Posisi
    ax.annotate(f'{round(p.get_height(),2)}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3),
                textcoords='offset points')
plt.xlabel(None)
plt.ylabel('None')
plt.xticks([1,0], ['Tidak Libur', 'Libur'])
plt.xticks(fontsize = 10)
plt.gca().ticklabel_format(style = 'plain', axis = 'y')
plt.yticks(fontsize = 10)
st.pyplot(plt)

st.subheader('Ratio Bike Rental Based On Weather for Two Years') #=================================================
weather = day_df.groupby('weathersit')['cnt'].mean().reset_index()
explode = (0, 0, 0.1)
plt.figure(figsize=(5, 5))
sns.set_color_codes('bright')
plt.pie(
    weather['cnt'], labels=weather['weathersit'], autopct='%1.1f%%', startangle=90, explode=explode, shadow=True
    )
st.pyplot(plt)

st.subheader('Rasio Casual and Registered User for Two Years') #=================================================
plt.figure(figsize=(5, 5))
total_casual = sum(day_df['casual'])
total_registered = sum(day_df['registered'])
data = [total_casual, total_registered]
labels = ['Casual', 'Registered']
plt.pie(data, labels=labels, autopct='%1.1f%%', wedgeprops=dict(width=0.4, edgecolor='w'))
st.pyplot(plt)

st.subheader('Clustering Based on Number of Bike Rental for Two Years') #=================================================
plt.figure(figsize = (15,7))
day_df['cluster'] = 'Other'
day_df.loc[(day_df['cnt'] >= day_df['cnt'].min()) & (day_df['cnt'] < day_df['cnt'].quantile(0.25)), 'cluster'] = 'Very Low'
day_df.loc[(day_df['cnt'] >= day_df['cnt'].quantile(0.25)) & (day_df['cnt'] < day_df['cnt'].quantile(0.50)), 'cluster'] = 'Low'
day_df.loc[(day_df['cnt'] >= day_df['cnt'].quantile(0.50)) & (day_df['cnt'] < day_df['cnt'].quantile(0.75)), 'cluster'] = 'High'
day_df.loc[(day_df['cnt'] >= day_df['cnt'].quantile(0.75)) & (day_df['cnt'] <= day_df['cnt'].max()), 'cluster'] = 'Very High'

color = {
    'Very Low' : 'blue',
    'Low' : 'green',
    'High' : 'yellow',
    'Very High' : 'red'
}

scatter_plot = plt.scatter(day_df['hum'], day_df['temp'], c=day_df['cluster'].map(color))
plt.title('Clustering Based on the Number of Bike Rental')
plt.ylabel('Temperature')
plt.xlabel('Humidity')

legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{label}') for label, color in color.items()]
plt.legend(handles=legend_labels, loc = 'upper left')
st.pyplot(plt)
st.write('The `Very Low` label means the number of bicycles rented that day is in the first quantile')
st.write('The `Low` label means the number of bicycles rented that day is in the second quantile')
st.write('The `High` label means the number of bicycles rented that day is in the third quantile')
st.write('The `Very High` label means the number of bicycles rented that day is in the fourth quantile')
