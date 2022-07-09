import pickle
import folium
import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt 
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Aplicacion CM",page_icon="🦁")
st.title('Aplicacion de prueba')

st.write('**Datos de king country, USA (20xx a 20xx)**:')
data=pd.read_csv('Carlosdata.csv')
latt, longg = data['lat'].mean(), data['long'].mean()
st.dataframe(data)

st.write('**Casas en el periodo analisado:**','Disponibles {} casas'.format(data['id'].nunique()))

st.write('**Casa mas barata:**')
st.dataframe(data[data['price']==data['price'].min()])

st.write('**Casa mas cara:**')
st.dataframe(data[data['price']==data['price'].max()])

st.title('Filtros')
##forma de seleccionarlos los datos para el filtro
OptFiltro = st.multiselect(
     'Que quieres filtrar',
     ['Habitaciones', 'Baños', 'Metros cuadrados (Espacio habitable)','Pisos','Vista al mar','Indice de construccion','Condicion', "Año"],
     ['Habitaciones', 'Baños','Pisos', 'Metros cuadrados (Espacio habitable)','Vista al mar','Indice de construccion','Condicion', "Año"])

if 'Precios' in OptFiltro:
    Price = st.number_input('Precios',step=1,min_value=0,max_value=round(data['price'].max()))

if 'Habitaciones' in OptFiltro:
    Bedrooms = st.number_input('Habitaciones',step=1,min_value=0,max_value=round(data['bedrooms'].max()), value=3)
    st.write('**Numero de habitaciones:**', str(Bedrooms))
else:
    Bedrooms = 3 ### by default

if 'Baños' in OptFiltro:
    Bathrooms = st.number_input ('Baños',step=1,min_value=0,max_value=round(data['bathrooms'].max()), value=2)
    st.write('**Baños:**', str(Bathrooms))
else:
    Bathrooms = 2 ### by default

if 'Metros cuadrados (Espacio habitable)' in OptFiltro:
    Mt2 = st.number_input('Metros cuadrados (Espacio habitable)',step=1,min_value=0,max_value=round(data['sqft_living'].max()), value=1680)
    st.write('**Metros cuadrados (Espacio habitable):**', str(Mt2))
else:
    Mt2 = 1680 ### by default

if 'Pisos' in OptFiltro:
    Floors = st.number_input('Pisos',step=1,min_value=0,max_value=round(data['floors'].max()))
    st.write('**Pisos:**', str(Floors))
else:
    Floors = 1 ### by default

if 'Vista al mar' in OptFiltro:
    Waterfront= st.number_input('Vista al mar',step=1,min_value=0,max_value=round(data['waterfront'].max()), value=0)
    
    if Waterfront ==1:
        st.write('**Vista al mar:**','Si')
    elif Waterfront ==0:
        st.write('**Vista al mar:**','No')
else:
    Waterfront=0 ### by default

if 'Condicion' in OptFiltro:
    Cond = st.number_input('Condicion',step=1,min_value=0,max_value=round(data['condition'].max()), value=3)
    st.write('**Condicion:**', Cond)
else:
    Cond = 3 ### by default

if 'Indice de construccion' in OptFiltro:
    Grade = st.number_input('Indice de construccion',step=1,min_value=0,max_value=round(data['grade'].max()), value=8)
    
    if 0<=Grade<=3:
        st.write('**Indice de construccion:**','Sin construir')
    elif 4<=Grade<=6:
        st.write('**Indice de construccion:**','Construccion y diseño pobre')
    elif 7<=Grade<=10:
        st.write('**Indice de construccion:**','Construccion y diseño promedio')
    elif 11<=Grade<=13:
        st.write('**Indice de construccion:**','Construccion y diseño de alta calidad')
else:
    Grade=8 ### by default

if 'Año' in OptFiltro:
    year = st.number_input('Edad del inmueble',step=1,min_value=0,max_value=2015, value=28)
    st.write('**Edad del inmueble:**', year)
else:
    year = 28 ### by default

st.title('Info de las casas:')


if 'Precios' in OptFiltro:
	if Price>0:
		st.write('Hay {} casas con Valor igual a {}'.format(data[data['price']==Price].shape[0],Price))

if 'Habitaciones' in OptFiltro:
	if Bedrooms>0:
		st.write('Hay {} casas con {} Habitacion/es'.format(data[data['bedrooms']==Bedrooms].shape[0],Bedrooms))

if 'Baños' in OptFiltro:
	if Bathrooms>0:
		st.write('Hay {} casas con {} baño/s'.format(data[data['bathrooms']==Bathrooms].shape[0],Bathrooms))

if 'Metros cuadrados (Espacio habitable)' in OptFiltro:
	if Mt2>0:
		st.write('Hay {} casas con {} metros cuadrados habitables'.format(data[data['sqft_living']==Mt2].shape[0],Mt2))

if 'Pisos' in OptFiltro:
	if Floors>0:
		st.write('Hay {} casas con {} Piso/s construidos'.format(data[data['floors']==Floors].shape[0],Floors))

if 'Vista al mar' in OptFiltro:
	if Waterfront>0:
		st.write('Hay {} casas con vista al mar'.format(data[data['waterfront']==Waterfront].shape[0]))

if 'Condicion' in OptFiltro:
	if Cond>0:
		st.write('Hay {} casas con una condicion igual a {}'.format(data[data['condition']==Cond].shape[0],Cond))

if 'Indice de construccion' in OptFiltro:
	if Grade>0:
		st.write('Hay {} casas con un Indice de construccion igual a {}'.format(data[data['grade']==Grade].shape[0],Grade))

        
def model_predict(lista):
    vecc = np.array([lista]).reshape(-1, 1).T
    model = pickle.load(open('model.pkl', 'rb'))
    return model.predict(vecc)[0]

try:
    valor = model_predict(lista=[Bedrooms, Bathrooms, Mt2, Waterfront, Cond, Grade, year]
                     )
except:
    valor = model_predict(lista=[0, 0, 0, 0, 0, 0, 0]
                     )
if valor<0:
    valor=0
st.header(f"Precio estimado de la vivienda: ${abs(round(valor, 2))}")


col1, col2 = st.columns(2)
with col1:
    
    data1 = data.copy()
    data1['zipcode'] = data['zipcode'].astype(str)
    
    data1 = data1[data1.bathrooms==Bathrooms]
    data1 = data1[data1.bedrooms==Bedrooms]
    data1 = data1[data1.waterfront==Waterfront]
    data1 = data1[data1.floors>=Floors]
    
    st.header("Densidad de Casas disponibles acorde a los requerimientos del usuario.")
    data1['price/sqft'] = data1['price']/data1['sqft_living']
    data_aux = data1[['price/sqft','zipcode']].groupby('zipcode').mean().reset_index()
    custom_scale = (data_aux['price/sqft'].quantile((0,0.2,0.4,0.6,0.8,1))).tolist()
    
    mapa = folium.Map(location=[latt, longg], zoom_start=8)
    url2 = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/KingCount.geojson'
    folium.Choropleth(geo_data=url2, 
                        data=data_aux,
                        key_on='feature.properties.ZIPCODE',
                        columns=['zipcode', 'price/sqft'],
                        threshold_scale=custom_scale,
                        fill_color='YlGn',
                        highlight=True).add_to(mapa)
    folium_static(mapa)

def get_params():
    params = {'Habitaciones':['bedrooms', Bedrooms],
          'Baños':['bathrooms', Bathrooms],
          'Edad':['year_old', year+2015]
              
         }
    return params

col1, col2 = st.columns(2)
with col1:
    params = get_params()
    data2 = data.copy()
    for filtro in OptFiltro:
        if filtro in params:
            (llave, variable) = params[filtro]
            data2 = data2[data2[llave]==variable]

    data2['zipcode'] = data2['zipcode'].astype(str)
    data2['price/sqft'] = data2['price']/data2['sqft_living']
    
    data2 = data2[data2.bathrooms==Bathrooms]
    data2 = data2[data2.bedrooms==Bedrooms]
    data2 = data2[data2.waterfront==Waterfront]
    data2 = data2[data2.floors>=Floors]
    
    st.header("Ubicación y detalles de casas disponibles acorde a los requerimientos del cliente.")
    mapa = folium.Map(location=[latt, longg], zoom_start=9)
    markercluster = MarkerCluster().add_to(mapa)
    for nombre, fila in data2.iterrows():
        folium.Marker([fila['lat'],fila['long']],
                         popup = 'Fecha: {} \n {} habitaciones \n {} baños \n constuida en {} \n área de {} pies cuadrados \n Precio por pie cuadrado: {}'.format(
                         fila['date'],
                         fila['bedrooms'],
                         fila['bathrooms'],
                         fila['yr_built'], 
                         fila['sqft_living'], 
                         fila['price/sqft'])
          ).add_to(markercluster)
    folium_static(mapa)

col1, col2 = st.columns(2)
# col1 = st.columns([1])
with col1: 
    st.write('Evolución del precio por cantidad de habitaciones.')
    with sns.axes_style("darkgrid"):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(7,7)) # try different values
        fig = sns.boxplot(x='bedrooms',y='price',data=data,showfliers=False)
        fig.set_xlabel("Cantidad de Habitaciones", fontsize = 17)
        fig.set_ylabel("Precio (Millones de Dólares)", fontsize = 17)
        fig = fig.figure
        st.pyplot(fig)
        
with col2:
    st.write('Evolución del precio por cantidad de baños.')
    with sns.axes_style("darkgrid"):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(7,7)) # try different values
        fig = sns.boxplot(x='bathrooms',y='price',data=data,showfliers=False)
        fig.set_xlabel("Cantidad de Baños", fontsize = 17)
        fig.set_ylabel("Precio (Millones de Dólares)", fontsize = 17)
        fig = fig.figure
        st.pyplot(fig)
