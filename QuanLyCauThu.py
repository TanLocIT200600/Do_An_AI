#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("F:/players_20.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe().columns


# In[6]:


df = df[['short_name','age', 'height_cm', 'weight_kg', 'overall', 'potential',
'value_eur', 'wage_eur', 'international_reputation', 'weak_foot',
'skill_moves', 'release_clause_eur', 'team_jersey_number',
'contract_valid_until', 'nation_jersey_number', 'pace', 'shooting',
'passing', 'dribbling', 'defending', 'physic', 'gk_diving',
'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
'gk_positioning', 'attacking_crossing', 'attacking_finishing',
'attacking_heading_accuracy', 'attacking_short_passing',
'attacking_volleys', 'skill_dribbling', 'skill_curve',
'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
'movement_reactions', 'movement_balance', 'power_shot_power',
'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
'mentality_aggression', 'mentality_interceptions',
'mentality_positioning', 'mentality_vision', 'mentality_penalties',
'mentality_composure', 'defending_marking', 'defending_standing_tackle',
'defending_sliding_tackle', 'goalkeeping_diving',
'goalkeeping_handling', 'goalkeeping_kicking',
'goalkeeping_positioning', 'goalkeeping_reflexes']]


# In[7]:


df.head()


# In[8]:


df = df[df.overall > 86]
df


# In[9]:


pd.set_option('display.max_rows',70)
df.isnull().sum()


# In[10]:


df = df.fillna(df.mean())


# In[11]:


df.isnull().sum()


# In[12]:


names=df.short_name.tolist()
df=df.drop(['short_name'],axis=1)


# In[13]:


df.head()


# In[14]:


from sklearn import preprocessing
x = df.values # numpy array
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)


# In[15]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # 2D PCA for the plot
reduced = pd.DataFrame(pca.fit_transform(X_norm))


# In[16]:


from sklearn.cluster import KMeans
# specify the number of clusters
kmeans = KMeans(n_clusters=5)
# fit the input data
kmeans = kmeans.fit(reduced)
# get the cluster labels
labels = kmeans.predict(reduced)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()


# In[17]:


reduced['cluster'] = clusters
reduced['name'] = names
reduced.columns = ['x', 'y', 'cluster', 'name']
reduced.head()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


sns.set(style="white")
ax = sns.lmplot(x="x", y="y", hue='cluster', data = reduced, legend=False,
fit_reg=False, size = 10, scatter_kws={"s": 250})
texts = []
for x, y, s in zip(reduced.x, reduced.y, reduced.name):
    texts.append(plt.text(x, y, s))

ax.set(ylim=(-2, 2))
plt.tick_params(labelsize=15)
plt.xlabel("PC 1", fontsize = 10)
plt.ylabel("PC 2", fontsize = 10)
plt.show()


# In[2]:


#Import all libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly as pl
from plotly.offline import plot
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Import the dataset
d_20 = pd.read_csv("F:/players_20.csv",error_bad_lines=False)


# In[4]:


d_20.head()


# In[5]:


d_20.shape


# In[6]:


cols = list(d_20.columns)

print(cols)


# In[7]:


u_c = ['dob','sofifa_id','player_url','long_name','body_type','real_face','loaned_from','nation_position','nation_jersey_number']


# In[8]:


d_20 = d_20.drop(u_c,axis=1)

d_20.head()


# In[9]:


d_20['BMI'] = d_20['weight_kg'] / ((d_20['height_cm'] / 100)**2)

d_20.head()


# In[10]:


d_20[['short_name','player_positions']]


# In[11]:


#Distributing the player positions in different columns
new_player_position = d_20['player_positions'].str.get_dummies(sep = ', ').add_prefix('Position_')

new_player_position.head()


# In[12]:


#Concatenate the new created columns with the dataset
d_20 = pd.concat([d_20,new_player_position], axis = 1)

d_20.head()


# In[13]:


#Dropping the original position column to eliminate confusion
d_20 = d_20.drop('player_positions', axis = 1)
d_20.head()


# In[14]:


positions = ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb']


# In[15]:


d_20[positions]


# In[16]:


for i in positions :
    d_20[i] = d_20[i].str.split('+', n = 1, expand = True)[0]
    
d_20.head()


# In[17]:


#Filling the null values with 0 and converting the column into integer value
d_20[positions] = d_20[positions].fillna(0)

d_20[positions] = d_20[positions].astype(int)

d_20[positions]


# In[18]:


style = ['dribbling','defending','physic','passing','shooting','pace']


# In[19]:


#Filling the null values in the above columns with the median values
for i in style : 
    d_20[i] = d_20[i].fillna(d_20[i].median())


# In[20]:


d_20[style].isnull().sum()


# In[21]:


d_20[style] = d_20[style].astype(int)


# In[22]:


#Filling all the null values of the data set by "0"

d_20 = d_20.fillna(0)

d_20.isnull().sum()


# In[23]:


a = d_20['age']

fig = go.Figure()
fig.add_trace(
    go.Histogram(x=a,
                marker=dict(color='rgba(114, 186, 59, 0.5)'))
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x', yref='paper',
                        x0=a.mean(), y0=0, x1=a.mean(), y1=0.9, line={'dash': 'dash'}),
)
fig.show()
print("Skewness of age is", d_20['age'].skew())


# In[24]:


d_20.loc[d_20['age'] == d_20['age'].min()]


# In[25]:


d_20.loc[d_20['age'] == d_20['age'].max()]


# In[26]:


#The age VS overall Rating comparison
avo = sns.lineplot(d_20['age'], d_20['overall'], palette = 'Wistia')
plt.title('Age vs Overall', fontsize = 20)

plt.show()


# In[27]:


b = d_20['BMI']

fig = go.Figure()
fig.add_trace(
    go.Histogram(x=b,
                marker=dict(color='rgba(114, 18, 59, 0.5)'),
                )
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x', yref='paper',
                        x0=b.mean(), y0=0, x1=b.mean(), y1=0.9, line={'dash': 'dash'}),
)
fig.show()

print("Skewness of BMI is", d_20['BMI'].skew())


# In[28]:


#Player with the highest BMI
d_20.loc[d_20['BMI'] == d_20['BMI'].max()][['short_name', 'age', 'overall', 'BMI']]


# In[29]:


#Player with the lowest BMI
d_20.loc[d_20['BMI'] == d_20['BMI'].min()][['short_name', 'age', 'overall', 'BMI']]


# In[30]:


pie_chart1 = px.pie(d_20, names = 'preferred_foot', title = 'Preffered Foot to Shoot')
pie_chart1.show()
print("The average of overall scores of players who prefer Right foot is", round(d_20.loc[d_20['preferred_foot'] == 'Right']['overall'].mean(), 2))

print("The average of overall scores of players who prefer Left foot is", round(d_20.loc[d_20['preferred_foot'] == 'Left']['overall'].mean(), 2))


# In[31]:


pie_chart2 = px.pie(d_20, names = 'international_reputation', title = 'International Reputation')
pie_chart2.show()
d_20['international_reputation'].value_counts()


# In[32]:


d_20.loc[d_20['international_reputation'] == 5]


# In[33]:


d_20.loc[d_20['international_reputation'] == 1].head(10)


# In[34]:


scatter_plot = go.Figure(
data = go.Scatter(
    x = d_20 ['overall'],
    y = d_20 ['value_eur'],
    mode = 'markers',
    marker = dict(
    size = 10,
    color = d_20['age'],
    showscale = True
    ),
    text = d_20['short_name']
)
)

scatter_plot.update_layout(title = 'Scatter Plot Year 2020',
                   xaxis_title = 'Overall Rating',
                   yaxis_title = 'Value in EUR')
scatter_plot.show()


# In[35]:


d_20.loc[d_20['value_eur'] == d_20['value_eur']].head(5)[['short_name','value_eur', 'club','age', 'overall' ]]


# In[36]:


d_20.loc[d_20['value_eur'] == d_20['value_eur']].tail(5)[['short_name','value_eur', 'club','age', 'overall' ]]


# In[37]:


#3D Scatter Plot
scatter5 = px.scatter_3d(d_20.head(50), x = 'overall', y = 'age', z = 'value_eur', color = 'short_name')
scatter5.update_layout(title = 'Top 50 player Value comparison with age and overall rating')
scatter5.show()


# In[38]:


pie_chart2 = px.pie(d_20, names = 'work_rate', title = 'Work Rate')
pie_chart2.show()
d_20['work_rate'].value_counts()


# In[39]:


d_20.loc[d_20['work_rate'] == 'High/High'].head(10)[['short_name','work_rate','age','BMI','club','overall','value_eur']]


# In[40]:


#List of all the club names
Club =np.unique(d_20['club'])

#List of means of the overall ratings of the clubs
Club_mean = d_20.groupby(d_20['club'])['overall'].mean()


# In[41]:


scatter_plot = go.Figure(
data = go.Scatter(
    x = Club,
    y = Club_mean,
    mode = 'markers',
    marker = dict(
    size = 10,
    color = d_20['value_eur']    
    )
)
)

scatter_plot.update_layout(title = 'Mean Overall Rating of all teams',
                   xaxis_title = 'Clubs',
                   yaxis_title = 'Overall Rating')
scatter_plot.show()


# In[42]:


#Club with player from most different nations
d_20.groupby(['club'])['nationality'].nunique().sort_values(ascending = False).head()


# In[43]:


#Club with player from least different nations
d_20.groupby(['club'])['nationality'].nunique().sort_values(ascending = True).head()


# In[44]:


attacking = ['RW','LW','ST','CF','LS','RS','LF','RF']

piech1 = d_20.query('team_position in @attacking')

piechart1 = px.pie(piech1, names = 'team_position', color_discrete_sequence= px.colors.sequential.Magenta_r,
                  title = 'Pie Chart For Attacking Positions')
piechart1.show()


# In[45]:


midfielding = ['CAM','RCM','CDM','LDM','RM','LM','LCM','RDM','RAM','CM','LAM']

piech2 = d_20.query('team_position in @midfielding')

piechart2 = px.pie(piech2, names = 'team_position', color_discrete_sequence= px.colors.sequential.Mint_r,
                  title = 'Pie Chart For Midfield Positions')
piechart2.show()


# In[46]:


defending = ['LCB','RCB','LB','RB','CB','RWB','LWB']

piech3 = d_20.query('team_position in @defending')

piechart3 = px.pie(piech3, names = 'team_position', color_discrete_sequence= px.colors.sequential.Teal_r,
                  title = 'Pie Chart For Defensive Positions')
piechart3.show()


# In[47]:


def top_players (pos, value):
    col = str('Position_')+str.upper(pos)
    targ = d_20[(d_20[col]==1) & (d_20['value_eur'] <= value)][['short_name','age','overall','BMI','value_eur']].head(10)
    return targ


# In[48]:


top_players('lw',50000000)


# In[49]:


d_19 = pd.read_csv("F:/players_19.csv",error_bad_lines=False)
d_18 = pd.read_csv("F:/players_18.csv",error_bad_lines=False)
d_17 = pd.read_csv("F:/players_17.csv",error_bad_lines=False)
d_16 = pd.read_csv("F:/players_16.csv",error_bad_lines=False)
d_15 = pd.read_csv("F:/players_15.csv",error_bad_lines=False)


# In[50]:


attributes = ['dribbling','defending','physic','passing','shooting','pace','overall']


# In[52]:


def playergrow(name):
    nm20 = d_20[d_20.short_name.str.contains(name, regex = False)]
    nm19 = d_19[d_19.short_name.str.contains(name, regex = False)]
    nm18 = d_18[d_18.short_name.str.contains(name, regex = False)]
    nm17 = d_17[d_17.short_name.str.contains(name, regex = False)]
    nm16 = d_16[d_16.short_name.str.contains(name, regex = False)]
    nm15 = d_15[d_15.short_name.str.contains(name, regex = False)]
    
    scat20 = go.Scatterpolar(
        r = [nm20['dribbling'].values[0],  nm20['defending'].values[0],   nm20['physic'].values[0], 
             nm20['passing'].values[0],     nm20['shooting'].values[0],    nm20['pace'].values[0], 
             nm20['overall'].values[0]
            ]
      ,
        theta = attributes,
        fill = 'toself',
        name = '2020'
    )
    scat19 = go.Scatterpolar(
        r = [nm19['dribbling'].values[0],  nm19['defending'].values[0],   nm19['physic'].values[0], 
             nm19['passing'].values[0],     nm19['shooting'].values[0],    nm19['pace'].values[0], 
             nm19['overall'].values[0]
            ]
      ,
        theta = attributes,
        fill = 'toself',
        name = '2019'
    )
    scat18 = go.Scatterpolar(
        r = [nm18['dribbling'].values[0],  nm18['defending'].values[0],   nm18['physic'].values[0], 
             nm18['passing'].values[0],     nm18['shooting'].values[0],    nm18['pace'].values[0], 
             nm18['overall'].values[0]
            ]
      ,
        theta = attributes,
        fill = 'toself',
        name = '2018'
    )
    scat17 = go.Scatterpolar(
        r = [nm17['dribbling'].values[0],  nm17['defending'].values[0],   nm17['physic'].values[0], 
             nm17['passing'].values[0],     nm17['shooting'].values[0],    nm17['pace'].values[0], 
             nm17['overall'].values[0]
            ]
      ,
        theta = attributes,
        fill = 'toself',
        name = '2017'
    )
    scat16 = go.Scatterpolar(
        r = [nm16['dribbling'].values[0],  nm16['defending'].values[0],   nm16['physic'].values[0], 
             nm16['passing'].values[0],     nm16['shooting'].values[0],    nm16['pace'].values[0], 
             nm16['overall'].values[0]
            ]
      ,
        theta = attributes,
        fill = 'toself',
        name = '2016'
    )
    scat15 = go.Scatterpolar(
        r = [nm15['dribbling'].values[0],  nm15['defending'].values[0],   nm15['physic'].values[0], 
             nm15['passing'].values[0],     nm15['shooting'].values[0],    nm15['pace'].values[0], 
             nm15['overall'].values[0]
            ]
      ,
        theta = attributes,
        fill = 'toself',
        name = '2015'
    )
    
    plan = [scat20, scat19, scat18, scat17, scat16, scat15]
    lay = go.Layout(
        polar = dict(
            radialaxis = dict(
                visible = True,
                range = [0,100]
            )
        )
        ,
        showlegend = True,
        title = 'Comparison of {} during years in years 2015 to 2020'.format(name)
    )
    figure = go.Figure (data = plan, layout = lay)
    figure.show()


# In[53]:


x = playergrow('Neymar')
y = playergrow('L. Messi')
z = playergrow('Cristiano Ronaldo')


# In[54]:


pie5 = px.pie(d_20.head(50),names='club',title='Clubs of top 50 players')
pie5.show()


# In[55]:


def bar_diagram (field):
    plt.figure(dpi=125)
    sns.countplot(field,data=d_20.head(50))
    plt.xlabel(field)
    plt.ylabel('Count')
    plt.title('Distribution of Top 50 players according to {}'.format(field))
    plt.show()


# In[56]:


bar_diagram('team_jersey_number')


# In[57]:


bar_diagram('age')


# In[59]:


scatter_plot2 = go.Figure(
data = go.Scatter(
    x = d_20 ['BMI'],
    y = d_20 ['pace'].head(50),
    mode = 'markers',
    marker = dict(
    size = 10,
    color = d_20['overall'],
    showscale = True
    ),
    text = d_20['short_name']
)
)

scatter_plot2.update_layout(title = 'BMI of top 50 players with best pace',
                   xaxis_title = 'BMI',
                   yaxis_title = 'Pace Rating')
scatter_plot2.show()


# In[ ]:




