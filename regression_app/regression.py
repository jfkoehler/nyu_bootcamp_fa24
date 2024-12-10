import streamlit as st 
import pickle
import pandas as pd
### regression model
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

houses = fetch_openml(data_id = 43926)
data = houses.frame
X = data[['Gr_Liv_Area', 'Overall_Qual', 'Sale_Condition', 'Lot_Area']]
y = data['Sale_Price']
transformer = make_column_transformer((OneHotEncoder(), X.select_dtypes('category').columns.tolist()),
                                      remainder = 'passthrough')
model = LinearRegression()
pipeline = Pipeline([('transformer', transformer), ('model', model)])
pipeline.fit(X, y)






st.header('Regression App')

gr_area = st.number_input('What is the above ground living area:')
lot_area = st.slider('What is the total lot area:')
over_qual = st.selectbox('What was the quality?', 
                         ('Above_Average', 'Average', 'Good', 'Very_Good', 'Excellent', 'Below_Average', 'Fair', 'Poor', 'Very_Excellent', 'Very_Poor'))
sale_cond = st.selectbox("Condition at sale?",
                         ('Normal', 'Partial', 'Family', 'Abnorml', 'Alloca', 'AdjLand'))
#bring in our model
# with open('lr_model.pkl', 'rb') as f:
#     model = pickle.load(f)
    
X = pd.DataFrame({'Gr_Liv_Area': gr_area,
                  'Overall_Qual': over_qual,
                  'Sale_Condition': sale_cond,
                  'Lot_Area': lot_area}, index = [0])

pred = model.predict(X)
st.write(pred)
    
