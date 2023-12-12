import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# icon_img = Image.open('Iris.jpg') ## Change the img
st.set_page_config(
    page_title='Provision',
    # page_icon=icon_img,
    layout="wide",
    initial_sidebar_state='collapsed'
)

json_vers = [
 {
  "Client_num_compt": 5800001864,
  "Client-age": 30,
  "Client_débit": 19,
  "Client_mvt": 1787,
  "Client_etat_comptepargne": "no",
  "Client profession": "unemployed",
  "Client_sex": "Woman",
  "Client_revenu": 23.5,
  "Client_sect": "*",
  "Client_statut": "married",
  "Client_Type_op": " ",
  "cridit en cours": 79
 },
 {
  "Client_num_compt": 5800001905,
  "Client-age": 33,
  "Client_débit": 11,
  "Client_mvt": 4789,
  "Client_etat_comptepargne": "no",
  "Client profession": "services",
  "Client_sex": "Woman",
  "Client_revenu": 34.916666666666664,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001020,
  "Client-age": 35,
  "Client_débit": 16,
  "Client_mvt": 1350,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 41.75,
  "Client_sect": "private",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800002020,
  "Client-age": 30,
  "Client_débit": 3,
  "Client_mvt": 1476,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 33.083333333333336,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001952,
  "Client-age": 59,
  "Client_débit": 5,
  "Client_mvt": 0,
  "Client_etat_comptepargne": "no",
  "Client profession": "ingeneer",
  "Client_sex": "Woman",
  "Client_revenu": 55.8333333333333,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800002598,
  "Client-age": 35,
  "Client_débit": 23,
  "Client_mvt": 747,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 22.5,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 11
 },
 {
  "Client_num_compt": 5800000125,
  "Client-age": 36,
  "Client_débit": 14,
  "Client_mvt": 307,
  "Client_etat_comptepargne": "no",
  "Client profession": "self-employed",
  "Client_sex": "Woman",
  "Client_revenu": 37.5,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 22
 },
 {
  "Client_num_compt": 5800003265,
  "Client-age": 39,
  "Client_débit": 6,
  "Client_mvt": 147,
  "Client_etat_comptepargne": "no",
  "Client profession": "technician",
  "Client_sex": "Man",
  "Client_revenu": 29.916666666666668,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800004578,
  "Client-age": 41,
  "Client_débit": 14,
  "Client_mvt": 221,
  "Client_etat_comptepargne": "no",
  "Client profession": "entrepreneur",
  "Client_sex": "Man",
  "Client_revenu": 21.75,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001245,
  "Client-age": 43,
  "Client_débit": 17,
  "Client_mvt": -88,
  "Client_etat_comptepargne": "no",
  "Client profession": "services",
  "Client_sex": "Man",
  "Client_revenu": 49.166666666666664,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001356,
  "Client-age": 39,
  "Client_débit": 20,
  "Client_mvt": 9374,
  "Client_etat_comptepargne": "no",
  "Client profession": "services",
  "Client_sex": "Man",
  "Client_revenu": 32.25,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 10
 },
 {
  "Client_num_compt": 5800003265,
  "Client-age": 43,
  "Client_débit": 17,
  "Client_mvt": 264,
  "Client_etat_comptepargne": "yes",
  "Client profession": "ingeneer",
  "Client_sex": "Man",
  "Client_revenu": 62.25,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 5
 },
 {
  "Client_num_compt": 5800003356,
  "Client-age": 36,
  "Client_débit": 13,
  "Client_mvt": 1109,
  "Client_etat_comptepargne": "no",
  "Client profession": "technician",
  "Client_sex": "Woman",
  "Client_revenu": 56.916666666666664,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 2
 },
 {
  "Client_num_compt": 5800003268,
  "Client-age": 20,
  "Client_débit": 30,
  "Client_mvt": 502,
  "Client_etat_comptepargne": "no",
  "Client profession": "student",
  "Client_sex": "Woman",
  "Client_revenu": 25.666666666666668,
  "Client_sect": "private",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 25
 },
 {
  "Client_num_compt": 5800001286,
  "Client-age": 31,
  "Client_débit": 29,
  "Client_mvt": 360,
  "Client_etat_comptepargne": "no",
  "Client profession": "blue-collar",
  "Client_sex": "Man",
  "Client_revenu": 30.5262,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 23
 },
 {
  "Client_num_compt": 5800002248,
  "Client-age": 40,
  "Client_débit": 29,
  "Client_mvt": 194,
  "Client_etat_comptepargne": "yes",
  "Client profession": "management",
  "Client_sex": "Woman",
  "Client_revenu": 40.25,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 10
 },
 {
  "Client_num_compt": 5800004681,
  "Client-age": 56,
  "Client_débit": 27,
  "Client_mvt": 4073,
  "Client_etat_comptepargne": "yes",
  "Client profession": "technician",
  "Client_sex": "Woman",
  "Client_revenu": 39.916666666666664,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800004445,
  "Client-age": 37,
  "Client_débit": 20,
  "Client_mvt": 2317,
  "Client_etat_comptepargne": "yes",
  "Client profession": "ingeneer",
  "Client_sex": "Man",
  "Client_revenu": 57.583333333333336,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001254,
  "Client-age": 25,
  "Client_débit": 23,
  "Client_mvt": -221,
  "Client_etat_comptepargne": "no",
  "Client profession": "blue-collar",
  "Client_sex": "Man",
  "Client_revenu": 29.666666666666668,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800008885,
  "Client-age": 31,
  "Client_débit": 7,
  "Client_mvt": 132,
  "Client_etat_comptepargne": "no",
  "Client profession": "services",
  "Client_sex": "Woman",
  "Client_revenu": 22.5,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800004598,
  "Client-age": 38,
  "Client_débit": 18,
  "Client_mvt": 0,
  "Client_etat_comptepargne": "yes",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 47.166666666666664,
  "Client_sect": "private",
  "Client_statut": "divorced",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800002222,
  "Client-age": 42,
  "Client_débit": 19,
  "Client_mvt": 16,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Woman",
  "Client_revenu": 43.583333333333336,
  "Client_sect": "private",
  "Client_statut": "divorced",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001111,
  "Client-age": 44,
  "Client_débit": 12,
  "Client_mvt": 106,
  "Client_etat_comptepargne": "no",
  "Client profession": "services",
  "Client_sex": "Man",
  "Client_revenu": 29.5,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001002,
  "Client-age": 44,
  "Client_débit": 7,
  "Client_mvt": 93,
  "Client_etat_comptepargne": "yes",
  "Client profession": "entrepreneur",
  "Client_sex": "Man",
  "Client_revenu": 41.25,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 12
 },
 {
  "Client_num_compt": 5800007770,
  "Client-age": 26,
  "Client_débit": 30,
  "Client_mvt": 543,
  "Client_etat_comptepargne": "yes",
  "Client profession": "housemaid",
  "Client_sex": "Woman",
  "Client_revenu": 9.25,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 11
 },
 {
  "Client_num_compt": 5800009992,
  "Client-age": 41,
  "Client_débit": 20,
  "Client_mvt": 5883,
  "Client_etat_comptepargne": "yes",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 39.583333333333336,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 10
 },
 {
  "Client_num_compt": 5800002068,
  "Client-age": 55,
  "Client_débit": 5,
  "Client_mvt": 627,
  "Client_etat_comptepargne": "no",
  "Client profession": "blue-collar",
  "Client_sex": "Woman",
  "Client_revenu": 29.9166666666667,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800004003,
  "Client-age": 67,
  "Client_débit": 17,
  "Client_mvt": 696,
  "Client_etat_comptepargne": "no",
  "Client profession": "retired",
  "Client_sex": "Man",
  "Client_revenu": 47.416666666666664,
  "Client_sect": "*",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 41
 },
 {
  "Client_num_compt": 5800004008,
  "Client-age": 56,
  "Client_débit": 30,
  "Client_mvt": 784,
  "Client_etat_comptepargne": "no",
  "Client profession": "self-employed",
  "Client_sex": "Man",
  "Client_revenu": 39.916666666666664,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 14
 },
 {
  "Client_num_compt": 5800002005,
  "Client-age": 53,
  "Client_débit": 21,
  "Client_mvt": 105,
  "Client_etat_comptepargne": "no",
  "Client profession": "ingeneer",
  "Client_sex": "Man",
  "Client_revenu": 71.4166666666667,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 50
 },
 {
  "Client_num_compt": 5800001852,
  "Client-age": 68,
  "Client_débit": 14,
  "Client_mvt": 4189,
  "Client_etat_comptepargne": "no",
  "Client profession": "retired",
  "Client_sex": "Man",
  "Client_revenu": 26.666666666666668,
  "Client_sect": "*",
  "Client_statut": "divorced",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001444,
  "Client-age": 31,
  "Client_débit": 27,
  "Client_mvt": 171,
  "Client_etat_comptepargne": "yes",
  "Client profession": "technician",
  "Client_sex": "Man",
  "Client_revenu": 35.75,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800004224,
  "Client-age": 59,
  "Client_débit": 21,
  "Client_mvt": 42,
  "Client_etat_comptepargne": "yes",
  "Client profession": "management",
  "Client_sex": "Woman",
  "Client_revenu": 42.583333333333336,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800006664,
  "Client-age": 32,
  "Client_débit": 26,
  "Client_mvt": 2536,
  "Client_etat_comptepargne": "yes",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 34.833333333333336,
  "Client_sect": "private",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800006464,
  "Client-age": 49,
  "Client_débit": 13,
  "Client_mvt": 1235,
  "Client_etat_comptepargne": "no",
  "Client profession": "technician",
  "Client_sex": "Man",
  "Client_revenu": 31.75,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800007474,
  "Client-age": 42,
  "Client_débit": 14,
  "Client_mvt": 1811,
  "Client_etat_comptepargne": "no",
  "Client profession": "ingeneer",
  "Client_sex": "Woman",
  "Client_revenu": 66.9166666666667,
  "Client_sect": "private",
  "Client_statut": "divorced",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800008899,
  "Client-age": 78,
  "Client_débit": 22,
  "Client_mvt": 229,
  "Client_etat_comptepargne": "yes",
  "Client profession": "retired",
  "Client_sex": "Man",
  "Client_revenu": 24.916666666666668,
  "Client_sect": "*",
  "Client_statut": "divorced",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800002424,
  "Client-age": 32,
  "Client_débit": 14,
  "Client_mvt": 2089,
  "Client_etat_comptepargne": "no",
  "Client profession": "blue-collar",
  "Client_sex": "Man",
  "Client_revenu": 27.833333333333332,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800003536,
  "Client-age": 33,
  "Client_débit": 6,
  "Client_mvt": 3935,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Man",
  "Client_revenu": 25.583333333333332,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001797,
  "Client-age": 23,
  "Client_débit": 30,
  "Client_mvt": 363,
  "Client_etat_comptepargne": "no",
  "Client profession": "services",
  "Client_sex": "Woman",
  "Client_revenu": 36.166666666666664,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800007849,
  "Client-age": 38,
  "Client_débit": 17,
  "Client_mvt": 11971,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Woman",
  "Client_revenu": 33.916666666666664,
  "Client_sect": "private",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 25
 },
 {
  "Client_num_compt": 5800004053,
  "Client-age": 36,
  "Client_débit": 11,
  "Client_mvt": 553,
  "Client_etat_comptepargne": "no",
  "Client profession": "management",
  "Client_sex": "Woman",
  "Client_revenu": 29.333333333333332,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800006418,
  "Client-age": 52,
  "Client_débit": 13,
  "Client_mvt": 1117,
  "Client_etat_comptepargne": "yes",
  "Client profession": "ingeneer",
  "Client_sex": "Woman",
  "Client_revenu": 95.8333333333333,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800005500,
  "Client-age": 32,
  "Client_débit": 13,
  "Client_mvt": 396,
  "Client_etat_comptepargne": "yes",
  "Client profession": "technician",
  "Client_sex": "Man",
  "Client_revenu": 36.666666666666664,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800008181,
  "Client-age": 32,
  "Client_débit": 21,
  "Client_mvt": 2204,
  "Client_etat_comptepargne": "yes",
  "Client profession": "technician",
  "Client_sex": "Woman",
  "Client_revenu": 32.666666666666664,
  "Client_sect": "private",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800001956,
  "Client-age": 34,
  "Client_débit": 7,
  "Client_mvt": 872,
  "Client_etat_comptepargne": "yes",
  "Client profession": "management",
  "Client_sex": "Woman",
  "Client_revenu": 38.916666666666664,
  "Client_sect": "public",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 12
 },
 {
  "Client_num_compt": 5800000058,
  "Client-age": 55,
  "Client_débit": 2,
  "Client_mvt": 145,
  "Client_etat_comptepargne": "no",
  "Client profession": "blue-collar",
  "Client_sex": "Man",
  "Client_revenu": 34.666666666666664,
  "Client_sect": "private",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800005001,
  "Client-age": 26,
  "Client_débit": 21,
  "Client_mvt": 0,
  "Client_etat_comptepargne": "no",
  "Client profession": "blue-collar",
  "Client_sex": "Woman",
  "Client_revenu": 26.5,
  "Client_sect": "public",
  "Client_statut": "married",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 },
 {
  "Client_num_compt": 5800005037,
  "Client-age": 32,
  "Client_débit": 4,
  "Client_mvt": -849,
  "Client_etat_comptepargne": "yes",
  "Client profession": "entrepreneur",
  "Client_sex": "Man",
  "Client_revenu": 129.083333333333,
  "Client_sect": "private",
  "Client_statut": "single",
  "Client_Type_op": "credit ",
  "cridit en cours": 0
 }
]
df = pd.read_json(json_vers)

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
    predicted_eligibility = y_pred[0]

    separation()
    st.markdown(
        f"Predicted Eligibility:<p style='font-family:Lucida Console;"
        f"text-align:center;"
        f"margin-top: -20px;"
        f"margin-left: 0%;"
        f"font-size:30px;"
        f"color:cyan;'>{predicted_eligibility}</p>",
        unsafe_allow_html=True
    )

    accuracy = accuracy_score(y_test, y_pred)
    st.text(f"The model's accuracy is : {accuracy}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='inferno',
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True']
                )
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix\n', fontsize=20)
    st.pyplot(plt)
