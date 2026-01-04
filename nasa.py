import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

@st.cache_resource
def load_model():
    return joblib.load('Disease_model.pkl')

@st.cache_data
def load_data():
        df = pd.read_csv(r"C:\Users\Admin\Downloads\Healthcare Symptoms–Disease Classification DatasetHealthcare Symptoms–Disease Classification Dataset\Healthcare.csv")
        df['Symptoms_List'] = df['Symptoms'].str.split(',')
        all_symptoms = sorted(set(sum(df['Symptoms_List'],[])))
        return df,all_symptoms



model = load_model()
df,all_symptoms = load_data()

for s in all_symptoms:
    df[s] = df['Symptoms_List'].apply(lambda x: int(s in x))
df['all_symptoms'] = df[all_symptoms].sum(axis = 1)

X = df[['Age'] + list(all_symptoms)] 
y = df['Disease']



st.title('Disease Classification')

st.sidebar.header("Filter Options")
gender_filter = st.sidebar.multiselect(
    "Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique()
)


age_range = st.slider('Select Age Range:',0,100,(10,50))
filter_data = df[
    (df['Gender'].isin(gender_filter)) & (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]


st.subheader("Filtered Data")
st.dataframe(filter_data)


st.subheader('Scatter Plot:symptom vs age')
fig,ax = plt.subplots(figsize = (10,6))

for gender in filter_data['Gender'].unique():
    gender_data = filter_data[filter_data['Gender'] == gender]
    ax.scatter(gender_data['Age'],gender_data['all_symptoms'],label = gender,alpha = 0.5)


ax.set_xlabel('Age')
ax.set_ylabel('Number of Symptoms')
ax.set_title('Age vs Number of Symptoms by gender')
ax.legend()
st.pyplot(fig)


st.subheader('Age vs Number of patient')
age_counts = filter_data.groupby('Age').size()
fig1,ax1 = plt.subplots()
ax1.bar(age_counts.index,age_counts.values)
ax1.set_xlabel('Age')
ax1.set_ylabel('Number of patients')
ax1.set_title('BarPlot')


st.pyplot(fig1)


st.subheader('Histogram')
fig2,ax2 = plt.subplots()

for gender in filter_data['Gender'].unique():
    gender_data = filter_data[filter_data['Gender'] == gender]
    ax2.hist(gender_data['Age'],bins = 10,alpha = 0.5,label = gender)

ax2.set_xlabel('Age')
ax2.set_ylabel('Count')
ax2.set_title('Histogram')
ax2.legend()
st.pyplot(fig2)

st.subheader('Boxplot')
fig3,ax3 = plt.subplots()
data = [filter_data[filter_data["Gender"] == g]['all_symptoms']
        for g in filter_data['Gender'].unique()
]
ax3.boxplot(data)
ax3.set_xticklabels(filter_data['Gender'].unique())
ax3.set_ylabel('Number of symptoms')
ax3.set_title('boxplot')
st.pyplot(fig3)


st.subheader('Corelation Map')
numeric_cols = filter_data.select_dtypes(include= 'number').columns
fig4,ax4 = plt.subplots(figsize = (10,6))

sns.heatmap(filter_data[numeric_cols].corr(),
            annot = True
            ,cmap = 'coolwarm',
            fmt = '.2f',
            linewidths= 0.5
            ,ax = ax4)
ax4.set_title('Corelation Map',fontsize = 14,pad=16)
plt.xticks(rotation = 45,ha = 'right')
plt.yticks(rotation = 0)
st.pyplot(fig4)




st.sidebar.header('Predict Disease')

age = st.sidebar.number_input('Age',1,90)
selected_symptoms = st.sidebar.multiselect('Select Symptoms',options = all_symptoms)


input_symptoms = [int(s in selected_symptoms)for s in all_symptoms]

if st.sidebar.button('Predict'):
    input_data = [[age] + input_symptoms]
    pred = model.predict(input_data)
    st.success(f'Predicted Disease:{pred[0]}')







                         


