import pickle
import time

import streamlit as st

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, accuracy_score

# matriz de confusion utilizando seaborn
def plot_confusion_matrix(cm):

    group_names = ['Verdadero Negativo','Falso Positivo','False Negativo','Verdadero Positivo']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_counts,group_names,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='BuPu')

    ax.set_xlabel('\nValores Predichos')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])

# curva auc-roc
def plot_auc_roc(model, x_test, y_test):
    y_pred_proba = model.predict(x_test) if selected_model == 3 else model.predict_proba(x_test)[::,1]        
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label="AUC="+str(auc))    

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc=4)

# curva de precision v recall para mostrar el tradeoff entre ambas métricas
def plot_precision_recall(model, x_test, y_test):
    y_pred_proba = model.predict(x_test) if selected_model == 3 else model.predict_proba(x_test)[::,1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    plt.plot(recall, precision, marker='.', label='Logistic Regression')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
        
    plt.legend()


# dictionaries for options
names = {
    1: 'Regresión Logística',
    2: 'SVM', 
    3: 'XGBoost'
}

captions = {
    1: "La regresión logística es un tipo de análisis utilizado para predecir el resultado de una variable categórica. Este tipo de análisis utiliza datos distribuidos binomialmente donde los números de ensayos Bernoulli son conocidos, mientras que las probabilidades de éxito son desconocidas. Este es un procedimiento cuantitativo muy útil para problemas donde la variable dependiente toma valores en un conjunto finito.",
    2: "Las máquinas de vectores de soporte son un conjunto de algoritmos de aprendizaje supervisado. Trabaja con datos etiquetados y entrena con datos históricos como entrada, para así aprender a asignar una etiqueta de salida. Este tipo de aprendizaje se utiliza para resolver problemas de clasificación y regresión.",
    3: "XGBoost es una implementación de una librería optimizada y distribuída de gradient boosting. Está diseñada para ser altamente eficiente, flexible y portable.",
}

models = {
    1: './models/log.pkl',
    2: './models/svm.pkl',
    3: './models/xgb.pkl',
}

cms = {
    1: './models/log_cm.pkl',
    2: './models/svm_cm.pkl',
    3: './models/xgb_cm.pkl',
}

metrics = {
    1: 'Matriz de Confusión',
    2: 'Curva ROC-AUC',
    3: 'Curva Precision-Recall',
    4: 'Reporte',
}

# sidebar configuration
selected_model = st.sidebar.selectbox("Elija un modelo", (1, 2, 3), format_func=lambda x: names[x])


# models and metrics
cm_filename    = cms[selected_model]
model_filename = models[selected_model]

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(cm_filename, 'rb') as file:
    cm = pickle.load(file)


# for predictions and probability predictions
with open('./models/x_test.pkl', 'rb') as file:
    x_test = pickle.load(file)
with open('./models/y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)


# pandas and data loading
data = pd.read_csv("./data/clean_data.csv")

important_columns = [
    'limit_bal', 'age', 'default_payment_next_month', 
    'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
    'female', 'education_1', 'education_2','education_3'
]

important_data = data[important_columns]


# raw data
st.title('Credit card default payment')
st.markdown('##### Predicción de default payment para clientes de tarjetas de crédito')

if st.checkbox('Mostrar información del dataset', value=True):
    st.subheader('Raw data')
    st.write(important_data)

# model
st.header(names[selected_model])
st.caption(captions[selected_model])

# eval metrics
selected_metric = st.selectbox("Métricas", (1, 2, 3, 4), format_func=lambda x: metrics[x])

fig = plt.figure(figsize=(10, 4))

if selected_metric < 4:
    if selected_metric == 1:
        plot_confusion_matrix(cm)
        
    elif selected_metric == 2:
        plot_auc_roc(model, x_test, y_test)
        
    elif selected_metric == 3:
        plot_precision_recall(model, x_test, y_test)

    st.pyplot(fig)
else:

    accuracy   = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    precision  = cm[0][0] / (cm[0][0] + cm[1][0])
    recall     = cm[0][0] / (cm[0][0] + cm[0][1])
    _precision = cm[1][1] / (cm[1][1] + cm[0][1])
    _recall    = cm[1][1] / (cm[1][1] + cm[1][0])
    
    st.markdown("##### Class 0")

    _, col2, col3 = st.columns(3)
    col2.metric("Precision", round(precision, 3), "-" if precision < 0.5 else "+")
    col3.metric("Recall", round(recall, 3), "-" if recall < 0.5 else "+")

    st.markdown("##### Class 1")

    _, col2, col3 = st.columns(3)
    col2.metric("Precision", round(_precision, 3), "-" if _precision < 0.5 else "+")
    col3.metric("Recall", round(_recall, 3), "-" if _recall < 0.5 else "+")

    st.markdown("##### Model")

    _, _, col3 = st.columns(3)
    col3.metric("Accuracy", round(accuracy, 3), "-" if accuracy < 0.5 else "+")

    with st.expander("Explicación"):
        st.markdown("* RECALL: Qué proporción de los elementos identificados en una clase es correcta.")
        st.markdown("* PRECISION: Cuántos de los verdaderos valores de la clase fueron identificados en ella.")
        st.markdown("* ACCURACY: Proporción de predicciones correctas hechas por el modelo")
        st.markdown("")

# using models to predict
st.header('Predicción')

st.markdown("Ingrese los datos solicitados para predecir si el cliente incurrirá o no en default payment")

st.markdown("##### Demográfico")
age = st.number_input("Edad", min_value=0, max_value=100, format="%d")

st.markdown("##### Montos de pago")
col1, col2, col3 = st.columns(3)
pay_amt4 = col1.number_input('Pago anterior en junio (dólares)', min_value=0.00)
pay_amt5 = col2.number_input('Pago anterior en mayo (dólares)', min_value=0.00)
pay_amt6 = col3.number_input('Pago anterior en abril (dólares)', min_value=0.00)

st.markdown("##### Estado de reembolso")
col1, col2, col3 = st.columns(3)
pay_1 = col1.number_input('Estado de reembolso septiembre', min_value=-1, max_value=9)
pay_2 = col2.number_input('Estado de reembolso agosto', min_value=-1, max_value=9)
pay_3 = col3.number_input('Estado de reembolso julio', min_value=-1, max_value=9)

col1, col2, col3 = st.columns(3)
pay_4 = col1.number_input('Estado de reembolso junio', min_value=-1, max_value=9,)
pay_5 = col2.number_input('Estado de reembolso mayo', min_value=-1, max_value=9,)
pay_6 = col3.number_input('Estado de reembolso abril', min_value=-1, max_value=9,)

# TODO: Traducir inputs a variables para predicción.
# TODO: Escalar las variables

predict = st.button("¡Hacer predicción!")

if predict:
    with st.spinner('Calculating...'):
        # TODO: Make predictions
        time.sleep(6)
        
    # TODO: Show result
    st.info('El cliente...')