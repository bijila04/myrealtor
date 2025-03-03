import glob
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import streamlit as st
from category_encoders import OneHotEncoder
from sklearn.linear_model import  Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split


def wrangle(path):
    df=pd.read_csv(path)
    size=[]
    for i in df['total_sqft']:
        x=i.split('-')
        if(len(x))==2:
            size.append((float(x[0])+float(x[1]))/2)
        else:
            try:
                size.append(float(i))
            except:
                size.append(None)

    #deal with outliers
    
    df['total_sqft']=size
    x,y=df['total_sqft'].quantile([0.1,0.9])
    mask=df['total_sqft'].between(x,y)
    df=df[mask]
    x,y=df['price'].quantile([0.1,0.9])
    mask=df['price'].between(x,y)
    df=df[mask]
    x,y=df['bath'].quantile([0.1,0.9])
    mask=df['bath'].between(x,y)
    df=df[mask]


      
   
    
    df.drop(columns=['area_type','availability','size','society','balcony'], inplace=True)
   
    df['price/sqft']=df['price']/df['total_sqft']
    return df


#Data
df1=wrangle('Bengaluru_House_Data.csv')
print(df1.describe())

st.title('My Realtor')

corr=df1['total_sqft'].corr(df1['bath']).round(2)

fig, ax = plt.subplots()

ax.scatter(df1['total_sqft'],df1['price'], color="green")
plt.title('Price Vs Area')
plt.xlabel('Price[Laks]')
plt.ylabel('Area[sqft]')

# Display in Streamlit


st.write(f'The Correlation of Price Vs Area is {corr}-as the area increases, the price tends to increase as well, but not perfectly.')
corr=df1['total_sqft'],df1['price']
print(corr)

st.pyplot(fig)

 # relationship of Price Vs Area

st.title('Group data by location to see if it has to do with prices')

df2=df1.groupby('location')['price/sqft'].mean().reset_index().sort_values(by='price/sqft',ascending=True)
df2=df2.set_index('location')
df2=df2.head(10)
fig = px.bar(df2, x="price/sqft", orientation='h', 
             title="House Prices by Region",text= 'price/sqft')
# Streamlit Display
st.plotly_chart(fig)

st.write('There is definetly some discrimination on prices based on the location of the property')



#get train data
df1.drop(columns=['price/sqft'], inplace=True) #remove this column coz it can be used to calculate the exact price 
features = ['location', 'total_sqft','bath']
target = 'price'
X=df1[features]
y=df1[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#get baseline
mean=y_train.mean()
y_baseline=[mean]* len(y_train)
mae= mean_absolute_error(y_train,y_baseline)
print(mae)

# make model
model= make_pipeline(
    OneHotEncoder('(use_cat_names=True)'),
    SimpleImputer(),
   
    Ridge()

)
model.fit(X_train,y_train)
#test using same data


y_test_predict=model.predict(X_test)
mae=round(mean_absolute_error(y_test,y_test_predict),2)
print(mae)
st.title('Predictor')
st.write(f'The model has a MAE of: {mae}')

options = sorted(list(df1['location'].unique()),key=str)

location = st.selectbox("Select Location:", options)
bathnum=st.slider('Number of Bathrooms',1,3)
area=st.number_input('Sqft',1000)
print(location)

def make_prediction(area,neighborhood,bathnum):
    data={
        'location':neighborhood,
        'total_sqft':area,
        'bath':bathnum,
    }
        
    df=pd.DataFrame(data,index=[0])
    
    prediction = model.predict(df).round(2)[0]
    return f"Predicted apartment price: ₹{prediction} Lakh"


if st.button("Predict!"):
    a=make_prediction(area,location,bathnum)
    st.write('Your house costs atleast', a)
    st.balloons()