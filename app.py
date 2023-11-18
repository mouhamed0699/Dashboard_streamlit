import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



df1=pd.read_csv('donnee/Sales_April_2019.csv')
df2=pd.read_csv('donnee/Sales_August_2019.csv')
df3=pd.read_csv('donnee/Sales_December_2019.csv')
df4=pd.read_csv('donnee/Sales_February_2019.csv')
df5=pd.read_csv('donnee/Sales_January_2019.csv')
df6=pd.read_csv('donnee/Sales_July_2019.csv')
df7=pd.read_csv('donnee/Sales_June_2019.csv')
df8=pd.read_csv('donnee/Sales_March_2019.csv')
df9=pd.read_csv('donnee/Sales_May_2019.csv')
df10=pd.read_csv('donnee/Sales_November_2019.csv')
df11=pd.read_csv('donnee/Sales_October_2019.csv')
df12=pd.read_csv('donnee/Sales_September_2019.csv')

df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12])

df.dropna(axis=0,inplace=True)

df.dropna(axis=0,inplace=True)

# Diviser l'adresse en adresse, ville et code postal
df["Ville"]=df["Purchase Address"].str.split(",",expand=True)[1]
df=df[df['Price Each']!='Price Each']

df['Price Each']=df['Price Each'].astype(float)
df['Quantity Ordered']=df['Quantity Ordered'].astype(float)

df['Montant vente']=df['Price Each']*df['Quantity Ordered']

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M')

df["Mois"]=df["Order Date"].dt.month.replace([1,2,3,4,5,6,7,8,9,10,11,12],
                                ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                                 )




st.set_page_config(page_title="Dashbord!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Tableau de Bord Ventes")

def Les_parametres():
    Ville = st.sidebar.selectbox('Ville', df['Ville'].unique())
    Product = st.sidebar.selectbox('Produit', df['Product'].unique())
    Mois = st.sidebar.selectbox('Mois', df['Mois'].unique())

    # Créer un dictionnaire avec les paramètres sélectionnés
    listeDict = [{'Ville': Ville, 'Product': Product, 'Mois': Mois}]

    return listeDict

# Exemple d'utilisation
st.sidebar.header('Filtres')

st.markdown(
    """
    <style>
        .css-6qob1r{
            background-color: paleturquoise;
        }
    </style>
    """,
    unsafe_allow_html=True
)

params = Les_parametres()

filtre_mois = df[df['Mois'] == params[0]['Mois']]
filtre_produit = df[df['Product'] == params[0]['Product']]
filtre_Ville = df[df['Ville'] == params[0]['Ville']]

chiffre_affaires_total = df['Montant vente'].sum()
chiffre_affaires_mois = filtre_mois['Montant vente'].sum()
quantites_par_produit = filtre_produit['Quantity Ordered'].sum()

# Affichage dans Streamlit


st.write("  ")

# create three columns
kpi1, kpi2, kpi3 = st.columns(3)
def format_number(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f} K"
    else:
        return f"{value:.0f}"

# Utilisation dans votre code
kpi1.metric(
    label="Chiffre d'affaires total 💰",
    value=format_number(chiffre_affaires_total),
    delta=None  # Remplacez cela par votre calcul de delta si nécessaire
)

kpi2.metric(
    label=f"Chiffre d'affaires de {params[0]['Mois']} 💹",
    value=format_number(chiffre_affaires_mois),
    delta=None  # Remplacez cela par votre calcul de delta si nécessaire
)

kpi3.metric(
    label=f"Quantité vendue pour {params[0]['Product']} 📦",
    value=format_number(quantites_par_produit),
    delta=None  # Remplacez cela par votre calcul de delta si nécessaire
)


st.markdown(
    """
    <style>
         .css-1wivap2 {
            color: brown ;  
        }

    </style>
    """,
    unsafe_allow_html=True
)

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Chiffre d'affaire"):
        
        st.write(f"Chiffre d'affaires total 💰: {chiffre_affaires_total:.0f} $", background_gradient="Blues")

        # Créer un DataFrame avec la valeur pour pouvoir le sauvegarder en CSV
        df_chiffre_affaires = pd.DataFrame({"Chiffre d'affaires total": [chiffre_affaires_total]})

        # Convertir le DataFrame en CSV
        csv_data = df_chiffre_affaires.to_csv(index=False).encode('utf-8')

        # Ajouter le bouton de téléchargement
        st.download_button("Download Data", data=csv_data, file_name="ChiffreAffairesTotal.csv", mime="text/csv",
                        help='Click here to download the data as a CSV file')

with cl2:
    with st.expander(f"Chiffre d'affaire de {params[0]['Mois']}"):
        st.write(f"Chiffre d'affaires de {params[0]['Mois']}  💰: {chiffre_affaires_mois:.0f} $", background_gradient="Blues")

        # Créer un DataFrame avec la valeur pour pouvoir le sauvegarder en CSV
        df_chiffre_affaires = pd.DataFrame({"Chiffre d'affaires total": [chiffre_affaires_mois]})

        # Convertir le DataFrame en CSV
        csv_data = df_chiffre_affaires.to_csv(index=False).encode('utf-8')

        # Ajouter le bouton de téléchargement
        st.download_button("Download Data", data=csv_data, file_name="ChiffreAffaires.csv", mime="text/csv",
                        help='Click here to download the data as a CSV file')

st.markdown(
    """
    <style>
         .st-dn{
            background: antiquewhite ;  
        }

    </style>
    """,
    unsafe_allow_html=True
)
st.write("  ")
fig_col1, fig_col2 = st.columns([2,3])

with fig_col1:
# Diagramme de ventes mensuelles
    st.subheader("Quantité vendu par produit")
    quantites_par_produit=df.groupby(df["Product"])['Quantity Ordered'].sum()
    quantites_par_produit = quantites_par_produit.reset_index()
    st.dataframe(quantites_par_produit)
    
with fig_col2:
    st.subheader("Quantité de Produit vendue par ville")
   # Diagramme de quantités vendues par ville
    fig_quantites_par_ville, ax_quantites_par_ville = plt.subplots(figsize=(20,10))
    df.groupby(df["Ville"])['Quantity Ordered'].sum().plot(kind='bar', ax=ax_quantites_par_ville, color='skyblue')
    ax_quantites_par_ville.set_title(f"")
    ax_quantites_par_ville.set_ylabel('Quantité Vendue')
    ax_quantites_par_ville.set_xlabel('Ville')
    ax_quantites_par_ville.yaxis.set_major_formatter('${x:,.2f}')
    st.pyplot(fig_quantites_par_ville)
    
    


fi_col1, fi_col2 = st.columns([2, 3])

with fi_col1:
    with st.expander("Quantité vendu par produit"):
        st.write(df.groupby(df["Product"])['Quantity Ordered'].sum(), background_gradient="Blues")


        quantites_par_produit = df.groupby(df["Product"])['Quantity Ordered'].sum()
        quantites_par_produit = quantites_par_produit.reset_index()
    
        # Créer un DataFrame avec les données pour le premier graphique
        df_quantites_produit = pd.DataFrame(quantites_par_produit)
        
        # Convertir le DataFrame en CSV
        csv_data_col1 = df_quantites_produit.to_csv(index=False).encode('utf-8')

        # Ajouter le bouton de téléchargement pour le premier graphique
        st.download_button("Download Data (Col1)", data=csv_data_col1, file_name="QuantitesProduit.csv", mime="text/csv",
                        help='Click here to download the data for Quantity Sold per Product as a CSV file')

with fi_col2:
    with st.expander("Quantité de Produit vendue par ville"):
        st.write(df.groupby(df["Ville"])['Quantity Ordered'].sum(), background_gradient="Blues")




        # Créer un DataFrame avec les données pour le deuxième graphique
        df_quantites_ville = df.groupby(df["Ville"])['Quantity Ordered'].sum().reset_index()
        
        # Convertir le DataFrame en CSV
        csv_data_col2 = df_quantites_ville.to_csv(index=False).encode('utf-8')

        # Ajouter le bouton de téléchargement pour le deuxième graphique
        st.download_button("Download Data (Col2)", data=csv_data_col2, file_name="QuantitesVille.csv", mime="text/csv",
                        help='Click here to download the data for Quantity Sold per City as a CSV file')



st.subheader("Tendances de Vente au Cours de l'Année")
fig, ax = plt.subplots(figsize=(20,6))
df.groupby(pd.Grouper(key='Order Date', freq='M'))['Montant vente'].sum().plot(ax=ax)
# ax.set_title(f"Montants de vente en fonction des mois")
ax.set_xlabel('Mois')
ax.set_ylabel('Montant de Vente (FCFA)')
ax.grid(True)
st.pyplot(fig)


with st.expander("Tendances de Vente au Cours de l'Année"):
    st.write(df.groupby(pd.Grouper(key='Order Date', freq='M'))['Montant vente'].sum(), background_gradient="Blues")




    # Créer un DataFrame avec les données pour le deuxième graphique
    df_tendance = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Montant vente'].sum().reset_index()
    
    # Convertir le DataFrame en CSV
    csv_data_col2 = df_tendance.to_csv(index=False).encode('utf-8')

    # Ajouter le bouton de téléchargement pour le deuxième graphique
    st.download_button("Download Data (Col2)", data=csv_data_col2, file_name="tendace.csv", mime="text/csv",
                    help='Click here to download ')



col1, col2 = st.columns(2)

with col1:
    
# Top 5 des produits vendus dans le mois sélectionné
    st.subheader(f" les 5 produits les plus vendus en {params[0]['Mois']}")
  
    top_5_produits = filtre_mois.groupby('Product')['Quantity Ordered'].sum().nlargest(5).index
    filtre_top_5 = filtre_mois[filtre_mois['Product'].isin(top_5_produits)]

    # Diagramme de quantités vendues des 5 produits les plus vendus
    fig_top_5_produits, ax_top_5_produits = plt.subplots(figsize=(20,10))
    sns.barplot(x='Product', y='Quantity Ordered', data=filtre_top_5, ax=ax_top_5_produits, color='blue')
    ax_top_5_produits.set_title("")
    ax_top_5_produits.set_ylabel('Quantité Vendue')
    ax_top_5_produits.set_xlabel('Produit')
    ax_top_5_produits.yaxis.set_major_formatter('${x:,.2f}')
    st.pyplot(fig_top_5_produits)

with col2:
    st.subheader(f"les 5 produits les plus vendus en {params[0]['Mois']} à {params[0]['Ville']}")
  
  
    # Top 5 des produits vendus dans la ville sélectionnée
    top_5_produits_par_ville = filtre_Ville.groupby('Product')['Quantity Ordered'].sum().nlargest(5).reset_index()

    # Diagramme de quantités vendues des 5 produits les plus vendus dans la ville sélectionnée
    fig_top_5_produits_ville, ax_top_5_produits_ville = plt.subplots(figsize=(20,10))
    sns.barplot(x='Quantity Ordered', y='Product', data=top_5_produits_par_ville, ax=ax_top_5_produits_ville, color='paleturquoise')
    ax_top_5_produits_ville.set_title("")
    ax_top_5_produits_ville.set_xlabel('Quantité Vendue')
    ax_top_5_produits_ville.set_ylabel('Produit')
    st.pyplot(fig_top_5_produits_ville)



coll1, coll2 = st.columns(2)

# Premier graphique
with coll1:
    with st.expander(f"Quantité vendue pour les 5 produits les plus vendus en {params[0]['Mois']}"):
        st.write(filtre_mois.groupby('Product')['Quantity Ordered'].sum().nlargest(5), background_gradient="Blues")

        st.subheader(f"Quantité vendue pour les 5 produits les plus vendus en {params[0]['Mois']}")
    
        top_5_produits = filtre_mois.groupby('Product')['Quantity Ordered'].sum().nlargest(5).index
        filtre_top_5 = filtre_mois[filtre_mois['Product'].isin(top_5_produits)]

        # Créer un DataFrame avec les données pour le premier graphique
        df_top_5_produits = filtre_top_5[['Product', 'Quantity Ordered']].copy()
        
        # Convertir le DataFrame en CSV
        csv_data_top_5_produits = df_top_5_produits.to_csv(index=False).encode('utf-8')

        # Ajouter le bouton de téléchargement pour le premier graphique
        st.download_button("Download Data (Top 5 Produits)", data=csv_data_top_5_produits, file_name="Top5Produits.csv", mime="text/csv",
                            help='Click here to download the data for Top 5 Sold Products as a CSV file')

# Deuxième graphique
with coll2:
    with st.expander(f"Quantité vendue pour les 5 produits les plus vendus en {params[0]['Mois']} à {params[0]['Ville']}"):
        st.write(filtre_Ville.groupby('Product')['Quantity Ordered'].sum().nlargest(5), background_gradient="Blues")

       

        # Top 5 des produits vendus dans la ville sélectionnée
        top_5_produits_par_ville = filtre_Ville.groupby('Product')['Quantity Ordered'].sum().nlargest(5).reset_index()


        df_top_5_produits_ville = top_5_produits_par_ville[['Product', 'Quantity Ordered']].copy()
        
        # Convertir le DataFrame en CSV
        csv_data_top_5_produits_ville = df_top_5_produits_ville.to_csv(index=False).encode('utf-8')

        # Ajouter le bouton de téléchargement pour le deuxième graphique
        st.download_button("Download Data (Top 5 Produits Ville)", data=csv_data_top_5_produits_ville, file_name="Top5ProduitsVille.csv", mime="text/csv",
                            help='Click here to download the data for Top 5 Sold Products in the selected City as a CSV file')




