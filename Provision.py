import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# icon_img = Image.open('Iris.jpg') ## Change the img
st.set_page_config(
    page_title='Provision',
    # page_icon=icon_img,
    layout="wide",
    initial_sidebar_state='collapsed'
)
filepath = r"Base_UIB.xlsx" # Download the data and change the path
df = pd.DataFrame(pd.read_excel(filepath))

df = df.reset_index().rename(
    columns={
        'Client_num_compt':'num_compte',
        'Client-age':'age',
        'Client_débit':'debit',
        'Client_mvt':'mvt',
        'Client_etat_comptepargne':'compte_epargne',
        'Client profession':'profession',
        'Client_sex':'sex',
        'Client_revenu':'revenu',
        'Client_sect':'secteur',
        'Client_statut':'statut',
        "Client_Typedop":'typed_op',
        'cridit en cours':'credit_en_cours'
        }
    )

df.drop('Client_Type_op', axis=1, inplace=True)
df.set_index('num_compte', inplace=True)
df['credit_class'] = df['revenu'].apply(lambda revenu : revenu >= 25.000)
df['credit_class']=df['credit_en_cours'].apply(lambda credit : credit==0)

label_encoder = LabelEncoder()
df["statut"]= label_encoder.fit_transform(df["statut"])
df["compte_epargne"]= label_encoder.fit_transform(df["compte_epargne"])
df["profession"]= label_encoder.fit_transform(df["profession"])
df["sex"]= label_encoder.fit_transform(df["sex"])
df["secteur"]= label_encoder.fit_transform(df["secteur"])


# Split data
X = df.drop('credit_class', axis=1)
y = df['credit_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=10)
model.fit(X_train, y_train)

def separation(titre1=None, titre2=None, color='cyan'):
  st.text('')
  if titre1:
    st.markdown(
      f"<h1 style='font-family: Lucida Console;text-align: center; color: {color};'>{titre1}</h1>",
      unsafe_allow_html=True
      )
  if titre2:
    st.markdown(
      f"<h4 style='font-family: Lucida Console;text-align: center; color: {color};'>{titre2}</h4>",
      unsafe_allow_html=True
      )
  st.text('')

separation('Provisionnement')
st.header("_:blue[Exploring the data]_")

revenu = st.slider(
  "Revenu",
  float(df['revenu'].min()),
  float(df['revenu'].max()),
  float(df['revenu'].mean())
)
credit_en_cours = st.slider(
  "Credit en Cours",
  float(df['credit_en_cours'].min()),
  float(df['credit_en_cours'].max()),
  float(df['credit_en_cours'].mean())
)


if st.button('Predict'):
  input_data = [[revenu, credit_en_cours]]
  y_pred = model.predict(input_data)
  predicted_eligibility = df.target_names[y_pred[0]]

  separation()
  st.mardown(
    f"Predicted Eligibility:<p style='font-family:Lucida Console;"
    f"text-align:center;"
    f"margin-top: -20px;"
    f"margin-left: 0%;"
    f"font-size:30px;"
    f"color:cyan;'>{predicted_eligibility}</p>",
    unsafe_allow_html=True
  )

  accuracy = accuracy_score(y_test, y_pred)
  st.text(st.text(f"The model's accuracy is : {accuracy}"))
  conf_matrix = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(conf_matrix,
              annot=True,
              fmt='d',
              cmap='inferno',
              xticklabels=['False', 'True'],
              yticklabels=['False', 'True']
              )
  plt.xlabel('Prédictions')
  plt.ylabel('Vraies valeurs')
  plt.title('Matrice de Confusion\n', fontsize=20)
  plt.show()
