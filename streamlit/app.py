import streamlit as st 
import pandas as pd
from sklearn.metrics import precision_score

st.header('Loan Prediction Contest')

st.write('Please enter your dataset of predictions below.  Make sure it is of the form:')

df = pd.DataFrame({'id': [1, 2, 3], 'prediction': [0, 0, 1]})

st.dataframe(df)

name = st.text_input('Your Name')
csv_file = st.file_uploader('Upload csv here')
solns = pd.read_csv('solution.csv')
if csv_file:
    ans_stu = pd.read_csv(csv_file)
    results = precision_score(solns['loan_status'], ans_stu.iloc[:, -1])
    st.write(solns.shape, ans_stu.shape)
    leaderboard = pd.DataFrame({'name': name, 'score': results})
    
# ex = pd.DataFrame({'name': ['Lenny'], 'score': [1.0]})
# ex.to_csv('res.csv', index = False)
    ans = pd.read_csv('res.csv')
    ans = pd.concat([ans, leaderboard], axis = 0)
    st.dataframe(ans)


