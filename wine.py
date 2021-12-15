import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, classification_report

st.set_page_config(page_title = "Wine Color and Score Classifier",
                   page_icon = ":wine_glass:",
                   layout = 'wide')

wine_data = pd.read_csv("Wine_Quality_Data.csv")

st.sidebar.image("https://i.imgur.com/ngvyFpY.png")
st.sidebar.title("")

navigation = st.sidebar.radio("Choix", ('Introduction','Data', 'Exploratory Data Analysis','Machine Learning Modeling', 'Analysez votre vin !'))

if navigation == 'Introduction' : 
    
    st.image("https://i.imgur.com/nkhoRDi.png")
    st.markdown("Cette application permet de classifier un vin en fonction de 11 caractéristiques afin de lui attribuer sa couleur (rouge ou blanc) ainsi qu'une notation allant de 1 à 10.")
    st.markdown("---")

    col1, col2 = st.columns(2)    
    
    with col1: 
        
        st.subheader("1 : Acidité Fixe")
        st.markdown("Il s'agit de l'acidité naturelle du raisin comme l'acide malique ou l'acide tartrique")
        
        st.subheader("2 : Acidité Volatile")
        st.markdown("L’acidité volatile d’un vin est constituée par la partie des acides gras comme l’acide acétique appartenant à la série des acides qui se trouvent dans le vin soit à l’état libre, soit à l’état salifié")

        st.subheader("3 : Acide Citrique")
        st.markdown("Acide présent surtout dans le citron où il intervient pour 95 % dans l’acidité de ce fruit. Il est aussi présent dans les raisins de tous les cépages. Sa concentration est plus élevée dans les moûts atteints de pourriture, ainsi que ceux dont les raisins sont issus du passerillage. Il  approche alors les 500 mg/l.")

        st.subheader("4 : Sucres résiduels")
        st.markdown("Les sucres résiduels (SR) sont les sucres (glucose + fructose)  encore présents dans le vin après fermentation. Ils ont donc été laissés intacts par les levures qui transforment les sucres (fermentescibles) en alcool.")


        st.subheader("5 : Chlorures")
        st.markdown("Matière minérale contenue naturellement dans le vin telle que le chlorure de sodium  (le sel) ainsi que d’autres chlorures (magnésium, calcium et potassium). Leur quantité dépend des conditions géographiques, géologiques et climatiques de culture de la vigne.")
    
    with col2 :   
        
        st.subheader("6 : Dioxyde de Soufre Total (TSO2)")
        st.markdown("C’est le fameux SO2 ou encore l’anhydride sulfureux connu aussi sous le code E 220. Le dioxyde de soufre est l’additif chimique le plus utilisé dans l’élaboration du vin. Si son utilisation aujourd’hui est très controversée, son apport est pourtant indispensable et bénéfique.")

        st.subheader("7 : Free Dioxyde de Soufre (FSO2)")
        st.markdown(" ")
                    
        st.subheader("8 : Densité des moùts")
        st.markdown("mesure de la teneur en sucre des moûts raisins ce qui permet de déterminer le degré alcoolique potentiel.")    

        st.subheader("9 : pH")
        st.markdown("Le potentiel hydrogène dit pH représente la force acide des vins. L’acidité totale correspond à la somme des acides présents sans notion de force.")   

        st.subheader("10 : Sulfate de cuivre")
        st.markdown("Le sulfate de cuivre est sans doute le plus célèbre fongicide utilisé pour le traitements de la vigne.")
        
        st.subheader("11 : Alcool")
        st.markdown("Tout simplement le niveau d'alcool du vin.")

    st.markdown("---")

scaler = StandardScaler()
MMS = MinMaxScaler()
    
X = wine_data.drop(['color','quality'], axis = 1)

X_SC = scaler.fit_transform(X)

X_SC_df = pd.DataFrame(X_SC, columns = X.columns)

X_MMS = MMS.fit_transform(X)

X_MMS_df = pd.DataFrame(X_MMS, columns = X.columns)

y_color = wine_data['color']

le = LabelEncoder()

y_color_LE = le.fit_transform(y_color)

y_quality = wine_data['quality']

description = wine_data.describe()


if navigation == 'Data' :
    
    st.image("https://i.imgur.com/Aj6xTns.png")
    
    st.subheader("Dataset original")
    expander = st.expander("Voir les données")
    with expander:
        clicked = st.dataframe(wine_data)
        shape = wine_data.shape
        st.markdown("Shape : " + str(shape))

    st.subheader("Dataset Descritpion")
    expander = st.expander("Voir les données")
    with expander:
        clicked = st.dataframe(description)

    
    st.subheader("X : Features")
    st.markdown("Qu'il s'agisse de la couleur ou de la qualité, nous utiliserons les mêmes 11 features décrites en Introduction")
    st.markdown("X : Features originales")
    expander = st.expander("Voir les données")
    with expander:
            clicked = st.dataframe(X)
            shape = X.shape
            st.markdown("Shape : " + str(shape))
            
    st.markdown("X_SC : Features standardisées avec StandardScaler")
    expander = st.expander("Voir les données")
    with expander:
            clicked = st.dataframe(X_SC_df)
            shape = X_SC.shape
            st.markdown("Shape : " + str(shape))
            
    st.markdown("X_MMS : Features standardisées avec MinMaxScaler")
    expander = st.expander("Voir les données")
    with expander:
            clicked = st.dataframe(X_MMS_df)
            shape = X_MMS.shape
            st.markdown("Shape : " + str(shape))
            
    st.subheader("Y : Targets")
            
    col1, col2 = st.columns(2)
        
    with col1 :
        st.markdown("y_color : Target pour déterminer la couleur")
        expander = st.expander("Voir les données")
        with expander:
            clicked = st.dataframe(wine_data['color'])
            shape = y_color.shape
            st.markdown("Shape : " + str(shape))
        st.markdown("y_color_LE : Target numerique pour déterminer la couleur")
        expander = st.expander("Voir les données")
        with expander:
            clicked = st.dataframe(y_color_LE)
            shape = y_color_LE.shape
            st.markdown("Shape : " + str(shape))
    
    with col2 : 
        st.markdown("y_quality : Target pour déterminer la qualité")
        expander = st.expander("Voir les données")
        with expander:
            clicked = st.dataframe(wine_data['quality'])
            shape = y_color.shape
            st.markdown("Shape : " + str(shape))


if navigation == 'Exploratory Data Analysis' :
    
    st.image("https://i.imgur.com/c76Q2Bx.png")
    
    navigation = st.selectbox("Navigation", options = ('1. Target Data Analysis', '2. Features Data Analysis','3. Correlation Analysis'))
    
    if navigation == '1. Target Data Analysis' : 
    
        st.subheader("Qualité")       
    
        col1, col2 = st.columns(2)
    
        with col1:
    
            boxplot_quality = px.box(wine_data, y = "quality")
        
            boxplot_quality.update_layout(
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   xaxis = dict(showgrid = False),
                   yaxis = dict(showgrid = False),
                   width = 300,
                   height = 350)
        
            boxplot_quality.update_traces(marker = dict(line = dict(color = '#FFFFFF', width = 1)))

            st.markdown("Boxplot")
            st.plotly_chart(boxplot_quality)
    
        with col2:
        
            quality_histo = px.histogram(wine_data, x = "quality")
        
            quality_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False),
                       bargap = 0.2)
        
            quality_histo.update_traces(marker = dict(line = dict(color = '#FFFFFF', width = 1)))
        
            st.markdown("Distribution")
            st.plotly_chart(quality_histo)
           
    
        
        st.subheader("Couleur")
    
        col1, col2 = st.columns(2)
    
        with col1:
        
            wine_data['counter'] = 1
        
            color_pie_chart = px.pie(wine_data, values = 'counter', names = 'color')
            
            color_pie_chart.update_layout(
                           width = 300,
                           height = 300,
                           showlegend = False)
            
            color_pie_chart.update_traces(textposition='inside',
                                             textinfo='percent+label',
                                             marker = dict(line = dict(color = '#FFFFFF', width = 1)),
                                             hole = 0.4)

            st.markdown("Pie Chart")
            st.plotly_chart(color_pie_chart)
    
        with col2:
        
            color_histo = px.histogram(wine_data, x = "color")
        
            color_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False),
                       bargap = 0.2)
        
            color_histo.update_traces(marker = dict(line = dict(color = '#FFFFFF', width = 1)))
        
            st.markdown("Distribution")
            st.plotly_chart(color_histo)

    if navigation == '2. Features Data Analysis' : 
        
        st.markdown("Qu'il s'agisse de déterminer la qualité ou la couleur du vin, les mêmes features seront utilisées. Il y n'y aura donc pas de distinction entre les features pour la couleur et les features pour la qualité")
        
        features_type = st.selectbox("Choisissez entre features originales ou features transformées", options = ('Originale','StandardScaler','MinMaxScaler'))
        
        if features_type == 'Originale':
            
            col1, col2, col3 = st.columns(3)
                
            with col1:
                    
                    fixed_acidity_histo = px.histogram(X, x = "fixed_acidity")
        
                    fixed_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Fixed Acidity')
                    st.plotly_chart(fixed_acidity_histo)
                    
                    residual_sugar_histo = px.histogram(X, x = "residual_sugar")
        
                    residual_sugar_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Residual Sugar')
                    st.plotly_chart(residual_sugar_histo)
                    
                    TSD_histo = px.histogram(X, x = "total_sulfur_dioxide")
        
                    TSD_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Total Sulfur Dioxide')
                    st.plotly_chart(TSD_histo)
                    
                    sulphates_histo = px.histogram(X, x = "sulphates")
                    
                    sulphates_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Sulphates')
                    st.plotly_chart(sulphates_histo)
                    
            with col2:
                    
                    volatile_acidity_histo = px.histogram(X, x = "volatile_acidity")
        
                    volatile_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Volatile Acidity')
                    st.plotly_chart(volatile_acidity_histo)
                    
                    chlorides_histo = px.histogram(X, x = "chlorides")
        
                    chlorides_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Chlorides')
                    st.plotly_chart(chlorides_histo)
                    
                    density_histo = px.histogram(X, x = "density")
        
                    density_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Density')
                    st.plotly_chart(density_histo)
                    
                    alcohol_histo = px.histogram(X, x = "alcohol")
                    
                    alcohol_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Alcohol')
                    st.plotly_chart(alcohol_histo)

            with col3:
                    
                    citric_acidity_histo = px.histogram(X, x = "citric_acid")
        
                    citric_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Citric Acid')
                    st.plotly_chart(citric_acidity_histo)
                    
                    FSD_histo = px.histogram(X, x = "free_sulfur_dioxide")
        
                    FSD_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Free Sulfure Dioxide')
                    st.plotly_chart(FSD_histo)

                    pH_histo = px.histogram(X, x = "pH")
        
                    pH_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('pH')
                    st.plotly_chart(pH_histo)

        if features_type == 'StandardScaler':
            
            col1, col2, col3 = st.columns(3)
                
            with col1:
                    
                    fixed_acidity_histo = px.histogram(X_SC_df, x = "fixed_acidity")
        
                    fixed_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Fixed Acidity')
                    st.plotly_chart(fixed_acidity_histo)
                    
                    residual_sugar_histo = px.histogram(X_SC_df, x = "residual_sugar")
        
                    residual_sugar_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Residual Sugar')
                    st.plotly_chart(residual_sugar_histo)
                    
                    TSD_histo = px.histogram(X_SC_df, x = "total_sulfur_dioxide")
        
                    TSD_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Total Sulfur Dioxide')
                    st.plotly_chart(TSD_histo)
                    
                    sulphates_histo = px.histogram(X_SC_df, x = "sulphates")
                    
                    sulphates_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Sulphates')
                    st.plotly_chart(sulphates_histo)
                    
            with col2:
                    
                    volatile_acidity_histo = px.histogram(X_SC_df, x = "volatile_acidity")
        
                    volatile_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Volatile Acidity')
                    st.plotly_chart(volatile_acidity_histo)
                    
                    chlorides_histo = px.histogram(X_SC_df, x = "chlorides")
        
                    chlorides_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Chlorides')
                    st.plotly_chart(chlorides_histo)
                    
                    density_histo = px.histogram(X_SC_df, x = "density")
        
                    density_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Density')
                    st.plotly_chart(density_histo)
                    
                    alcohol_histo = px.histogram(X_SC_df, x = "alcohol")
                    
                    alcohol_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Alcohol')
                    st.plotly_chart(alcohol_histo)

            with col3:
                    
                    citric_acidity_histo = px.histogram(X_SC_df, x = "citric_acid")
        
                    citric_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Citric Acid')
                    st.plotly_chart(citric_acidity_histo)
                    
                    FSD_histo = px.histogram(X_SC_df, x = "free_sulfur_dioxide")
        
                    FSD_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Free Sulfure Dioxide')
                    st.plotly_chart(FSD_histo)

                    pH_histo = px.histogram(X_SC_df, x = "pH")
        
                    pH_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('pH')
                    st.plotly_chart(pH_histo)

        if features_type == 'MinMaxScaler':
            
            col1, col2, col3 = st.columns(3)
                
            with col1:
                    
                    fixed_acidity_histo = px.histogram(X_MMS_df, x = "fixed_acidity")
        
                    fixed_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Fixed Acidity')
                    st.plotly_chart(fixed_acidity_histo)
                    
                    residual_sugar_histo = px.histogram(X_MMS_df, x = "residual_sugar")
        
                    residual_sugar_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Residual Sugar')
                    st.plotly_chart(residual_sugar_histo)
                    
                    TSD_histo = px.histogram(X_MMS_df, x = "total_sulfur_dioxide")
        
                    TSD_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Total Sulfur Dioxide')
                    st.plotly_chart(TSD_histo)
                    
                    sulphates_histo = px.histogram(X_MMS_df, x = "sulphates")
                    
                    sulphates_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Sulphates')
                    st.plotly_chart(sulphates_histo)
                    
            with col2:
                    
                    volatile_acidity_histo = px.histogram(X_MMS_df, x = "volatile_acidity")
        
                    volatile_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Volatile Acidity')
                    st.plotly_chart(volatile_acidity_histo)
                    
                    chlorides_histo = px.histogram(X_MMS_df, x = "chlorides")
        
                    chlorides_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Chlorides')
                    st.plotly_chart(chlorides_histo)
                    
                    density_histo = px.histogram(X_MMS_df, x = "density")
        
                    density_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Density')
                    st.plotly_chart(density_histo)
                    
                    alcohol_histo = px.histogram(X_MMS_df, x = "alcohol")
                    
                    alcohol_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Alcohol')
                    st.plotly_chart(alcohol_histo)

            with col3:
                    
                    citric_acidity_histo = px.histogram(X_MMS_df, x = "citric_acid")
        
                    citric_acidity_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Citric Acid')
                    st.plotly_chart(citric_acidity_histo)
                    
                    FSD_histo = px.histogram(X_MMS_df, x = "free_sulfur_dioxide")
        
                    FSD_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('Free Sulfure Dioxide')
                    st.plotly_chart(FSD_histo)

                    pH_histo = px.histogram(X_MMS_df, x = "pH")
        
                    pH_histo.update_layout(
                       width = 300,
                       height = 350,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       yaxis = dict(showgrid = False))
                    
                    st.markdown('pH')
                    st.plotly_chart(pH_histo)

    if navigation == '3. Correlation Analysis' : 
        
        correlation_target = st.selectbox('Choisissez la target avec laquelle vous voulez voir les correlations', options = ('Couleur','Qualité'))
        
        if correlation_target == 'Qualité' :
            
            correlation_quality_df = wine_data.drop(['color'], axis = 1)
            y = correlation_quality_df['quality']
            fields = list(correlation_quality_df.columns[:-1])
            correlations = correlation_quality_df[fields].corrwith(y)
            correlations.sort_values(inplace=True, ascending = False)
            correlations = pd.DataFrame(correlations)
            correlations['Feature'] = correlations.index
            correlations.columns = ['Correlation','Feature']
            correlations.reset_index(drop=True, inplace = True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('Correlation en valeur relative')
                bar_chart = px.bar(correlations, x = 'Correlation', y = 'Feature')
                bar_chart.update_layout(
                           width = 400,
                           height = 400,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           yaxis = dict(showgrid = False, autorange = "reversed"),
                           xaxis = dict(showgrid = False))
                st.plotly_chart(bar_chart)
                
                expander = st.expander("Voir les données")
                with expander:
                    clicked = st.dataframe(correlations)
            
            with col2:
                
                correlations_abs = correlations
                correlations_abs['Correlation'] = correlations_abs['Correlation'].abs()
                correlation_abs = correlations_abs.sort_values(by = ['Correlation'], inplace = True)
            
                st.markdown('Correlation en valeur absolue')
                bar_chart = px.bar(correlations_abs, x = 'Correlation', y = 'Feature')
                bar_chart.update_layout(
                           width = 400,
                           height = 400,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           yaxis = dict(showgrid = False),
                           xaxis = dict(showgrid = False))
                st.plotly_chart(bar_chart)
                
                expander = st.expander("Voir les données")
                with expander:
                    clicked = st.dataframe(correlations_abs)
            
        if correlation_target == 'Couleur' :
            
            correlation_color_df = wine_data.drop(['quality'], axis = 1)
            correlation_color_df['color'] = le.fit_transform(correlation_color_df['color'])
            y = correlation_color_df['color']
            fields = list(correlation_color_df.columns[:-1])
            correlations = correlation_color_df[fields].corrwith(y)
            correlations.sort_values(inplace=True, ascending = False)
            correlations = pd.DataFrame(correlations)
            correlations['Feature'] = correlations.index
            correlations.columns = ['Correlation','Feature']
            correlations.reset_index(drop=True, inplace = True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('Correlation en valeur relative')
                bar_chart = px.bar(correlations, x = 'Correlation', y = 'Feature')
                bar_chart.update_layout(
                           width = 400,
                           height = 400,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           yaxis = dict(showgrid = False, autorange = "reversed"),
                           xaxis = dict(showgrid = False))
                st.plotly_chart(bar_chart)
                
                expander = st.expander("Voir les données")
                with expander:
                    clicked = st.dataframe(correlations)
            
            with col2:
                
                correlations_abs = correlations
                correlations_abs['Correlation'] = correlations_abs['Correlation'].abs()
                correlation_abs = correlations_abs.sort_values(by = ['Correlation'], inplace = True)
            
                st.markdown('Correlation en valeur absolue')
                bar_chart = px.bar(correlations_abs, x = 'Correlation', y = 'Feature')
                bar_chart.update_layout(
                           width = 400,
                           height = 400,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           yaxis = dict(showgrid = False),
                           xaxis = dict(showgrid = False))
                st.plotly_chart(bar_chart)
                
                expander = st.expander("Voir les données")
                with expander:
                    clicked = st.dataframe(correlations_abs)
                    

# ML Color
                    
X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(X, y_color_LE, test_size=0.4, random_state=42)
X_train_color_SC, X_test_color_SC, y_train_color_SC, y_test_color_SC = train_test_split(X_SC, y_color_LE, test_size=0.4, random_state=42)
X_train_color_MMS, X_test_color_MMS, y_train_color_MMS, y_test_color_MMS = train_test_split(X_MMS, y_color_LE, test_size=0.4, random_state=42) 

X_train_quality, X_test_quality, y_train_quality, y_test_quality = train_test_split(X, y_quality, test_size=0.4, random_state=42, stratify= y_quality)
X_train_quality_SC, X_test_quality_SC, y_train_quality_SC, y_test_quality_SC = train_test_split(X_SC, y_quality, test_size=0.4, random_state=42, stratify= y_quality)
X_train_quality_MMS, X_test_quality_MMS, y_train_quality_MMS, y_test_quality_MMS = train_test_split(X_MMS, y_quality, test_size=0.4, random_state=42, stratify= y_quality)

@st.cache(persist = True)
def ML_reports(classifier, X_train, y_train, X_test, y_test):
                
    model = classifier

    model.fit(X_train, y_train)
                
    predictions = model.predict(X_test)
                
    report = classification_report(y_test, predictions, output_dict = True)
                
    report_df = pd.DataFrame(report).transpose()
                
    return report_df

                    
if navigation == 'Machine Learning Modeling' :
    
    st.image('https://i.imgur.com/Jpqk67m.png')
    st.markdown('Pour analyser votre vin, nous avons entrainé et optimisé les hyperparamètres de 3 modèles de Machine Learning de Classification sur 2 targets différentes à savoir la couleur et la qualité.')
 
    col1, col2, col3 = st.columns(3)
    

    with col1:
        st.markdown('1. Support Vector Machine (SVM)')
        
    with col2:
        st.markdown('2. KNearest-Neighbors (KNN)')
        
    with col3:
        st.markdown('3. Decision Tree')
    
    ML_model = st.selectbox("Choisissez un modèle pour voir le processus d'optimisation", options = ('1. Support Vector Machine (SVM)','2. KNearest-Neighbors (KNN)','3. Decision Tree'))
    
    if ML_model == '1. Support Vector Machine (SVM)' : 
        
        targets = st.selectbox('Choisissez une target', options = ('Couleur', 'Qualité'))

        
        if targets == 'Couleur' : 
            
            model = SVC()
            
            st.subheader('Déterminer la couleur du vin avec SVM')
            st.markdown('Pour commencer, nous avons simplement entrainé le modèle sans modifier les hyperparamètres, voici les rapports obtenus selon si les features sont standardisées ou non :')
        
            st.markdown('X (features non standardisées)')
            st.markdown('')
            X_SVC_REPORT = ML_reports(model, X_train_color, y_train_color, X_test_color, y_test_color)
            st.dataframe(X_SVC_REPORT)

            st.markdown('X_SC (features standardisées avec StandardScaler)')
            X_SC_SVC_REPORT = ML_reports(model, X_train_color_SC, y_train_color_SC, X_test_color_SC, y_test_color_SC)
            st.dataframe(X_SC_SVC_REPORT)

            st.markdown('X_MMS (features standardisées avec MinMaxScaler)')
            X_MMS_SVC_REPORT = ML_reports(model, X_train_color_MMS, y_train_color_MMS, X_test_color_MMS, y_test_color_MMS)
            st.dataframe(X_MMS_SVC_REPORT)
        
            st.markdown('Nous remarquons que les meilleurs resultats se trouvent avec les features standardisées avec StandardScaler. Comme les résultats sont très satisfaisants, nous ne ferons pas de tuning pour les hyperparameters.')
            st.markdown('Conclusion : Pour déterminer la couleur avec SVM nous utiliserons les features X_SC et ne feront pas de tuning des hyperparameters.')
            
        if targets == 'Qualité' : 
            
            model = SVC(C = 10, gamma = 1)
            
            st.subheader('Déterminer la qualité du vin avec SVM')
            st.markdown("Les résultats sans faire du tuning d'hyperparamètres n'étaient pas satisfaisants, donc nous avons cherché les hyperparamètres optimaux grâce à GridSearchCV de Scikit Learn.")
            st.markdown("Les paramètres utilisés pour le GridSearchCV sont les suivants:")
            st.code('''param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}''', language = 'python')
            st.markdown("Il en resort que le best_estmator est :")
            st.code('''SVC(C=10, gamma=1)''', language = 'python')
            st.markdown('Voici les rapports de précision avec le best_estimator : ')
            
            st.markdown('X (features non standardisées)')
            st.markdown('')
            X_SVC_REPORT = ML_reports(model, X_train_quality, y_train_quality, X_test_quality, y_test_quality)
            st.dataframe(X_SVC_REPORT)

            st.markdown('X_SC (features standardisées avec StandardScaler)')
            X_SC_SVC_REPORT = ML_reports(model, X_train_quality_SC, y_train_quality_SC, X_test_quality_SC, y_test_quality_SC)
            st.dataframe(X_SC_SVC_REPORT)

            st.markdown('X_MMS (features standardisées avec MinMaxScaler)')
            X_MMS_SVC_REPORT = ML_reports(model, X_train_quality_MMS, y_train_quality_MMS, X_test_quality_MMS, y_test_quality_MMS)
            st.dataframe(X_MMS_SVC_REPORT)
            
        

    if ML_model == '2. KNearest-Neighbors (KNN)' : 
        
        targets = st.selectbox('Choisissez une target', options = ('Couleur', 'Qualité'))
        
        if targets == 'Couleur' : 
            
            model = KNeighborsClassifier()
            
            st.subheader('Déterminer la couleur du vin avec KNN')
            st.markdown('Pour commencer, nous avons simplement entrainé le modèle sans modifier les hyperparamètres, voici les rapports obtenus selon si les features sont standardisées ou non ')
        
            st.markdown('X (features non standardisées)')
            X_KNN_REPORT = ML_reports(model, X_train_color, y_train_color, X_test_color, y_test_color)
            st.dataframe(X_KNN_REPORT)

            st.markdown('X_SC (features standardisées avec StandardScaler)')
            X_SC_KNN_REPORT = ML_reports(model, X_train_color_SC, y_train_color_SC, X_test_color_SC, y_test_color_SC)
            st.dataframe(X_SC_KNN_REPORT)

            st.markdown('X_MMS (features standardisées avec MinMaxScaler)')
            X_MMS_KNN_REPORT = ML_reports(model, X_train_color_MMS, y_train_color_MMS, X_test_color_MMS, y_test_color_MMS)
            st.dataframe(X_MMS_KNN_REPORT)

            st.markdown('Nous remarquons que les resultats avec les features standardisées par StandardScaler et MinMaxScaler sont très proches. Comme les résultats sont très satisfaisants, nous ne ferons pas de tuning pour les hyperparameters.')
            st.markdown('Conclusion : Pour déterminer la couleur avec KNN nous utiliserons les features X_SC et ne feront pas de tuning des hyperparameters.')
            
        if targets == 'Qualité' : 
            
            model = KNeighborsClassifier(metric='manhattan', n_neighbors=18, weights='distance')
            
            st.subheader('Déterminer la qualité du vin avec KNN')
            st.markdown("Les résultats sans faire du tuning d'hyperparamètres n'étaient pas satisfaisants, donc nous avons cherché les hyperparamètres optimaux grâce à GridSearchCV de Scikit Learn.")
            st.markdown("Les paramètres utilisés pour le GridSearchCV sont les suivants:")
            st.code('''param_grid = {'n_neighbors': list(range(1,30)),
              'weights': ['uniform','distance'],
              'metric': ['euclidean','manhattan']}''', language = 'python')
            st.markdown("Il en resort que le best_estmator est :")
            st.code('''KNeighborsClassifier(metric='manhattan', n_neighbors=18, weights='distance')''', language = 'python')
            st.markdown('Voici les rapports de précision avec le best_estimator : ')
            
            st.markdown('X (features non standardisées)')
            st.markdown('')
            X_KNN_REPORT = ML_reports(model, X_train_quality, y_train_quality, X_test_quality, y_test_quality)
            st.dataframe(X_KNN_REPORT)

            st.markdown('X_SC (features standardisées avec StandardScaler)')
            X_SC_KNN_REPORT = ML_reports(model, X_train_quality_SC, y_train_quality_SC, X_test_quality_SC, y_test_quality_SC)
            st.dataframe(X_SC_KNN_REPORT)

            st.markdown('X_MMS (features standardisées avec MinMaxScaler)')
            X_MMS_KNN_REPORT = ML_reports(model, X_train_quality_MMS, y_train_quality_MMS, X_test_quality_MMS, y_test_quality_MMS)
            st.dataframe(X_MMS_KNN_REPORT)


    if ML_model == '3. Decision Tree' : 
        
        targets = st.selectbox('Choisissez une target', options = ('Couleur', 'Qualité'))
             
        if targets == 'Couleur' : 

            model = DecisionTreeClassifier()
            
            st.subheader('Déterminer la couleur du vin avec DecisionTree')
            st.markdown('Pour commencer, nous avons simplement entrainé le modèle sans modifier les hyperparamètres, voici les rapports obtenus selon si les features sont standardisées ou non ')
        
            st.markdown('X (features non standardisées)')
            X_DT_REPORT = ML_reports(model, X_train_color, y_train_color, X_test_color, y_test_color)
            st.dataframe(X_DT_REPORT)

            st.markdown('X_SC (features standardisées avec StandardScaler)')
            X_SC_DT_REPORT = ML_reports(model, X_train_color_SC, y_train_color_SC, X_test_color_SC, y_test_color_SC)
            st.dataframe(X_SC_DT_REPORT)

            st.markdown('X_MMS (features standardisées avec MinMaxScaler)')
            X_MMS_DT_REPORT = ML_reports(model, X_train_color_MMS, y_train_color_MMS, X_test_color_MMS, y_test_color_MMS)
            st.dataframe(X_MMS_DT_REPORT)

            st.markdown('Nous remarquons que les meilleurs resultats se trouvent avec les features standardisées avec MinMaxScaler. Comme les résultats sont très satisfaisants, nous ne ferons pas de tuning pour les hyperparameters.')
            st.markdown('Conclusion : Pour déterminer la couleur avec DecisionTree nous utiliserons les features X_MMS et ne feront pas de tuning des hyperparameters.')     

        if targets == 'Qualité' : 
            
            model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=75)
            
            st.subheader('Déterminer la qualité du vin avec Decision Tree')
            st.markdown("Les résultats sans faire du tuning d'hyperparamètres n'étaient pas satisfaisants, donc nous avons cherché les hyperparamètres optimaux grâce à GridSearchCV de Scikit Learn.")
            st.markdown("Les paramètres utilisés pour le GridSearchCV sont les suivants:")
            st.code('''param_grid = {'max_depth': np.arange(0,100, 5).tolist(),
              'min_samples_leaf': np.arange(0,100, 5).tolist(),
              'criterion': ['entropy','gini']}''', language = 'python')
            st.markdown("Il en resort que le best_estmator est :")
            st.code('''DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=75)''', language = 'python')
            st.markdown('Voici les rapports de précision avec le best_estimator : ')
            
            st.markdown('X (features non standardisées)')
            st.markdown('')
            X_DT_REPORT = ML_reports(model, X_train_quality, y_train_quality, X_test_quality, y_test_quality)
            st.dataframe(X_DT_REPORT)

            st.markdown('X_SC (features standardisées avec StandardScaler)')
            X_SC_DT_REPORT = ML_reports(model, X_train_quality_SC, y_train_quality_SC, X_test_quality_SC, y_test_quality_SC)
            st.dataframe(X_SC_DT_REPORT)

            st.markdown('X_MMS (features standardisées avec MinMaxScaler)')
            X_MMS_DT_REPORT = ML_reports(model, X_train_quality_MMS, y_train_quality_MMS, X_test_quality_MMS, y_test_quality_MMS)
            st.dataframe(X_MMS_DT_REPORT)


if navigation == 'Analysez votre vin !' :
    
    st.image("https://i.imgur.com/7IyaXgn.png")
    st.markdown("Pour déterminer la couleur et la qualité du vin nous utiliserons, pour chaqun des trois modèles, la meilleure combinaison de features / hyperparameters déterminée dans la partie précédente.")
    
    st.markdown("Veuillez entrer les caractéristiques de votre vin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity (entre 1 et 20)", min_value = 1.0, max_value = 20.0, value = 7.0)
        residual_sugar = st.number_input("Residual Sugar (entre 0 et 100)", min_value = 0.0, max_value = 100.0, value = 5.0)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (entre 0 et 500)", min_value = 0.0, max_value = 500.0, value = 100.0)
        sulphates = st.number_input("Sulphates (entre 0 et 2)", min_value = 0.0, max_value = 5.0, value = 0.5)
    
    with col2:
        volatile_acidity = st.number_input("Volatile Acidity (entre 0 et 2)", min_value = 0.0, max_value = 2.0, value = 0.3)
        chlorides = st.number_input("Chlorides (entre 0 et 100)", min_value = 0.0, max_value = 1.0, value = 0.05)
        densty = st.number_input("Density (entre 0 et 2)", min_value = 0.0, max_value = 2.0, value = 1.0)
        alcohol = st.number_input("Alcohol (entre 8 et 15)", min_value = 0.0, max_value = 15.0, value = 10.0)
        
    with col3:
        citric_acid = st.number_input("Citric Acid (entre 0 et 2)", min_value = 0.0, max_value = 2.0, value = 0.3)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (entre 0 et 300)", min_value = 0.0, max_value = 300.0, value = 30.0)
        pH = st.number_input("pH (entre 0 et 5)", min_value = 0.0, max_value = 5.0, value = 3.0)

    X_input = np.array([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,densty,pH,sulphates,alcohol])

    if st.button("Analyser votre vin !"):
        
        col1, col2, col3 = st.columns(3)
        
        with col1: 
        
            st.subheader("Support Vector Machine")
    
            model = SVC()
        
            model.fit(X_train_color,y_train_color)
        
            predicted_color_input = model.predict(X_input.reshape(1,-1))
            
            if predicted_color_input == 0 : 
                st.markdown("Couleur prédite : [rouge]")
            
            if predicted_color_input == 1 : 
                st.markdown("Couleur prédite : [blanc]")
            
            model = SVC(C = 10, gamma = 1)
        
            model.fit(X_train_quality,y_train_quality)
        
            predicted_quality_input = model.predict(X_input.reshape(1,-1))
        
            st.markdown("Qualité prédite : " + str(predicted_quality_input))
        
            if predicted_color_input == 0 and predicted_quality_input == 3 :
            
                st.image("https://i.imgur.com/g1EinZ6.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 4 :
            
                st.image("https://i.imgur.com/r1amE4B.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 5 :
            
                st.image("https://i.imgur.com/bSKFh9o.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 6 :
            
                st.image("https://i.imgur.com/ufbyPNZ.png")

            if predicted_color_input == 0 and predicted_quality_input == 7 :
            
                st.image("https://i.imgur.com/XSUyJTm.png")

            if predicted_color_input == 0 and predicted_quality_input == 8 :
            
                st.image("https://i.imgur.com/9GB1eoy.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 9 :
            
                st.image("https://i.imgur.com/k1FSCyn.png")

            if predicted_color_input == 1 and predicted_quality_input == 3 :
            
                st.image("https://i.imgur.com/fVe4Jl1.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 4 :
            
                st.image("https://i.imgur.com/x81g947.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 5 :
            
                st.image("https://i.imgur.com/7ysqnqd.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 6 :
            
                st.image("https://i.imgur.com/6kVBfx6.png")

            if predicted_color_input == 1 and predicted_quality_input == 7 :
            
                st.image("https://i.imgur.com/m37yfYm.png")

            if predicted_color_input == 1 and predicted_quality_input == 8 :
            
                st.image("https://i.imgur.com/ryPwB0s.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 9 :
            
                st.image("https://i.imgur.com/OXRUCq8.png")

        with col2: 
        
            st.subheader("KNN")
            st.title(" ")
    
            model = KNeighborsClassifier()
        
            model.fit(X_train_color,y_train_color)
        
            predicted_color_input = model.predict(X_input.reshape(1,-1))
        
            if predicted_color_input == 0 : 
                st.markdown("Couleur prédite : [rouge]")
            
            if predicted_color_input == 1 : 
                st.markdown("Couleur prédite : [blanc]")
            
            model = KNeighborsClassifier(metric='manhattan', n_neighbors=18, weights='distance')
         
            model.fit(X_train_quality,y_train_quality)
        
            predicted_quality_input = model.predict(X_input.reshape(1,-1))
        
            st.markdown("Qualité prédite : " + str(predicted_quality_input))
        
            if predicted_color_input == 0 and predicted_quality_input == 3 :
            
                st.image("https://i.imgur.com/g1EinZ6.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 4 :
            
                st.image("https://i.imgur.com/r1amE4B.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 5 :
            
                st.image("https://i.imgur.com/bSKFh9o.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 6 :
            
                st.image("https://i.imgur.com/ufbyPNZ.png")

            if predicted_color_input == 0 and predicted_quality_input == 7 :
            
                st.image("https://i.imgur.com/XSUyJTm.png")

            if predicted_color_input == 0 and predicted_quality_input == 8 :
            
                st.image("https://i.imgur.com/9GB1eoy.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 9 :
            
                st.image("https://i.imgur.com/k1FSCyn.png")

            if predicted_color_input == 1 and predicted_quality_input == 3 :
            
                st.image("https://i.imgur.com/fVe4Jl1.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 4 :
            
                st.image("https://i.imgur.com/x81g947.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 5 :
            
                st.image("https://i.imgur.com/7ysqnqd.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 6 :
            
                st.image("https://i.imgur.com/6kVBfx6.png")

            if predicted_color_input == 1 and predicted_quality_input == 7 :
            
                st.image("https://i.imgur.com/m37yfYm.png")

            if predicted_color_input == 1 and predicted_quality_input == 8 :
            
                st.image("https://i.imgur.com/ryPwB0s.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 9 :
            
                st.image("https://i.imgur.com/OXRUCq8.png")


        with col3: 
        
            st.subheader("Decision Tree")
            st.title(" ")
    
            model = DecisionTreeClassifier()
        
            model.fit(X_train_color,y_train_color)
        
            predicted_color_input = model.predict(X_input.reshape(1,-1))
        
            if predicted_color_input == 0 : 
                st.markdown("Couleur prédite : [rouge]")
            
            if predicted_color_input == 1 : 
                st.markdown("Couleur prédite : [blanc]")
            
            model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=75)
        
            model.fit(X_train_quality,y_train_quality)
        
            predicted_quality_input = model.predict(X_input.reshape(1,-1))
        
            st.markdown("Qualité prédite : " + str(predicted_quality_input))
        
            if predicted_color_input == 0 and predicted_quality_input == 3 :
            
                st.image("https://i.imgur.com/g1EinZ6.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 4 :
            
                st.image("https://i.imgur.com/r1amE4B.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 5 :
            
                st.image("https://i.imgur.com/bSKFh9o.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 6 :
            
                st.image("https://i.imgur.com/ufbyPNZ.png")

            if predicted_color_input == 0 and predicted_quality_input == 7 :
            
                st.image("https://i.imgur.com/XSUyJTm.png")

            if predicted_color_input == 0 and predicted_quality_input == 8 :
            
                st.image("https://i.imgur.com/9GB1eoy.png")
                
            if predicted_color_input == 0 and predicted_quality_input == 9 :
            
                st.image("https://i.imgur.com/k1FSCyn.png")

            if predicted_color_input == 1 and predicted_quality_input == 3 :
            
                st.image("https://i.imgur.com/fVe4Jl1.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 4 :
            
                st.image("https://i.imgur.com/x81g947.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 5 :
            
                st.image("https://i.imgur.com/7ysqnqd.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 6 :
            
                st.image("https://i.imgur.com/6kVBfx6.png")

            if predicted_color_input == 1 and predicted_quality_input == 7 :
            
                st.image("https://i.imgur.com/m37yfYm.png")

            if predicted_color_input == 1 and predicted_quality_input == 8 :
            
                st.image("https://i.imgur.com/ryPwB0s.png")
                
            if predicted_color_input == 1 and predicted_quality_input == 9 :
            
                st.image("https://i.imgur.com/OXRUCq8.png")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
