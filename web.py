# Importar librerías
import streamlit as st
import pandas as pd
import joblib

# Cargar los modelos
def loadModels():
    rf_model = joblib.load('rf_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    return rf_model, lr_model
    
def loadPipeline():
    return joblib.load('pipeline.pkl')

# Funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'No Superviviente'
    elif num == 1:
        return 'Superviviente'

def user_input_parameters():
    pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    age = st.sidebar.slider('Age', 1.0, 70.0, 1.0)
    sibsp = st.sidebar.selectbox('SibSp', [0,1])
    parch = st.sidebar.selectbox('Parch', [0,1])
    fare = st.sidebar.slider('Fare', 0.0, 500.0, 0.0)
    embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
    }
    features = pd.DataFrame(data, index=[0])
    return features

    
def main():
    st.title('Modelamiento de Examen Final')
    st.sidebar.header('Parámetros de entrada al usuario')
    
    rf_model, lr_model = loadModels()
    
    df = user_input_parameters()
    pipeline = loadPipeline()
    encoded = pipeline.transform(df)
    data = pd.DataFrame(encoded, columns=['nominal__Sex_1', 'nominal__Sex_2', 'nominal__Embarked_1',
       'nominal__Embarked_2', 'nominal__Embarked_3', 'remainder__Pclass',
       'remainder__Age', 'remainder__SibSp', 'remainder__Parch',
       'remainder__Fare'])
    #escoger el modelo preferido
    option = ['Logistic Regression', 'Random Forest']
    model = st.sidebar.selectbox('Which model you like to use?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)

    if st.button('RUN'):
        if model == 'Logistic Regressor':
            st.success(classify(lr_model.predict(data)))
        else:
            st.success(classify(rf_model.predict(data)))

if __name__ == '__main__':
    main()
    