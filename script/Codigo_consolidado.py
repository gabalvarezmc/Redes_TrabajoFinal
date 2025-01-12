#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Adolfo Fuentes </em><br>
# 
# </div>
# 
# ### **Tarea Final. Ciencia de Redes**
# 
# *11 de Enero de 2025*
# 
# #### Integrantes: 
# ` Gabriel Álvarez, Giuseppe Lavarello, Ingrid Solís, Rosario Valderrama `

# # Objetivos:

# Se trabajará con los datos de servicios y destinos turísticos de la región de los Ríos. El objetivo principal del trabajo es construir, analizar y extraer información valiosa de una red bipartita basada en datos de visitas turísticas de la región de Los Ríos, con un enfoque en la inferencia de patrones significativos.
# 
# El análisis de redes es una herramienta poderosa para, en este caso, comprender y modelar las relaciones complejas entre usuarios y destinos turísticos. 
# 
# Para el análisis, será importante considerar lo siguiente:
# 
# #### Análisis Exploratorio de Datos:
# 
# - Explorar y describir el conjunto de datos utilizando herramientas estadísticas y visuales: histogramas, analizar relaciones bivariadas entre variables relevantes, calcular y visualizazr correlaciones entre las variables.
# 
# #### Inferencia de la Red:
# 
# - Construir una red bipartita que conecte usuarios y destinos turísticos, donde los enlaces reflejen interacciones como visitas o calificaciones.
# 
# - Generar proyecciones unimodales de la red bipartita, para analizar usuarios conectados por destintos compartidos o analizar destinos conectados por  usuarios comunes.
# 
# - Aplicar modelos nulos para filtrar relaciones débiles, asegurando que las conexiones retenidas sean estadísticamente significativas.
# 
# #### Descripción de la Red:
# 
# - Caracterizar la red original y las redes proyectadas mediante:
# 
#       - Distribución de grados antes y después de filtrar
#       - Calcular el coeficiente de clustering para evaluar la cohesión de la red
#       - Medir densidad y número de nodos y enlaces de la red
#   
# - Detectar comunidades dentro de las redes proyectadas utilizando algoritmos de modularidad, analizando sus características y posibles implicancias.
# 
# #### Identificación de Patrones en la Red:
# 
# - Identificar patrones relevantes en la red proyectada, como nodos clave (hubs) o centralidad, relaciones frecuentes entre destinos turísticos que indiquen co-visitas significativas.
# 
# - Analizar las comunidades detectadas para entender preferencias compartidas entre usuarios.
# 
# #### Objetivo Final
# 
# - Evaluar cómo los patrones y métricas extraídos de la red pueden aplicarse en la creación de sistemas de recomendación personalizados para los usuarios.
# 
# - Proponer estrategias basadas en los hallazgos para optimizar la gestión turística, fomentar la diversificación de destinos y promover la sostenibilidad en el sector.
# 

# # Motivación

# <div style="text-align: justify;">
# Anexo: Contexto y Motivación de Datos de Destinos y Servicios Turísticos:  
# El turismo es un motor esencial para el desarrollo sostenible, impulsando el progreso hacia los Objetivos de Desarrollo Sostenible (ODS) de las Naciones Unidas a través del crecimiento económico, la preservación cultural y la gestión ambiental. No obstante, la pandemia de COVID-19 afectó profundamente a la industria turística global, creando desafíos sin precedentes en toda la cadena de valor. Aunque se ha observado una recuperación parcial, los niveles de actividad turística previos a la pandemia aún no se han alcanzado completamente.<br><br>
# 
# En este escenario, la industria turística de Chile destaca por su significativa contribución a la economía nacional y por sus logros en turismo sostenible y de aventura. Reconocido internacionalmente en los World Travel Awards 2023, Chile se ha posicionado como un destino preferido para viajeros eco-conscientes y aventureros. Sin embargo, para mantener y potenciar esta ventaja competitiva en la era pospandemia, es crucial adoptar tecnologías innovadoras que mejoren y personalicen las experiencias turísticas.<br><br>
# 
# La transformación digital y la integración de la inteligencia artificial (IA) permiten a las empresas turísticas obtener profundos insights sobre las preferencias y comportamientos de los viajeros. Los sistemas de recomendación, alimentados por reseñas y calificaciones en línea, juegan un papel crucial al influir en las decisiones de los clientes y mejorar los servicios ofrecidos. Estos sistemas personalizados no solo aumentan la satisfacción del cliente, sino que también promueven prácticas turísticas responsables al alinearse con los valores y motivaciones de los viajeros.<br><br>
# 
# Construir una red de Destinos Turísticos y de Servicios turísticos nos permite mapear qué actividades económicas relacionadas tienden a agruparse, lo que es esencial para comprender los patrones de clustering y co-ubicación o co-visita de los servicios turísticos. Un análisis de esta naturaleza puede contribuir al desarrollo económico y social de las comunidades locales, alineándose con los ODS y fortaleciendo la posición de Chile como líder en turismo sostenible. </div>
# 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import nltk
import string
from itertools import compress
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import networkx as nx


# ## 1. Análisis Exploratorio de Datos:

df_base = pd.read_excel("./data/consolidado_reviews_Los_Rios.xlsx").reset_index()
df_base = df_base.rename({'index':'Id'}, axis='columns')


df_base


# - **Id**: es el identificador único asignado a cada review de la base de datos. **En total hay 479.343 reviews**
# - **N**: es el identificador numérico único asignado a cada lugar que cuenta con reviews en Google Maps.
# - **place_id**: es el identificador único asignado a cada lugar que cuenta con reviews en Google Maps.
# - **name**: es el nombre único que corresponde a cada lugar que cuenta con al menos una review en Google Maps.
# - **id_review**: corresponde a la identificación única de la review realizada en Google Maps.
# - **user name**: es el nombre del usuario que realiza al menos una review en Google Maps.
# - **id_usuario**: es el identificador único asignado a cada usuario que ha realizado al menos un review en Google Maps.
# - **stars**: es la valoración general del lugar en una escala de estrellas en Google Maps, donde 1 es la más baja y 5 es la más alta.
# - **link_review**: es el link que lleva a la review realizada por un usuario en Google Maps.
# - **fecha**: es la fecha en que se realizó la review en Google Maps.
# - **review**: es el texto que realiza una persona para comentar su experiencia en un lugar específico en Google Maps.

df_base.head(2)


df_base.columns


print('Cantidad de Reviews: '+str(len(df_base)))
print('Cantidad de Comentarios: ' + str(len(df_base['review'].dropna())))
print('Cantidad de Valoraciones: ' + str(len(df_base['stars'].dropna())))
print('Diferencia (Valoraciones - Comentarios): ' + str(len(df_base['stars'].dropna()) - len(df_base['review'].dropna())))
print('Cantidad de Lugares: '+ str(len(df_base['place_id'].unique())))
print('Cantidad de usuarios únicos: '+str(len(df_base['id_usuario'].unique())))


# En esta base de datos se encuentra **479,343 reviews**, de las cuales se tiene **233,588** reviews con comentarios y **245,755** reviews solo con valoración de estrellas (sin comentarios). Los usuarios únicos son **205,836** y los lugares valorizados en Google Maps son **1,390**.  
# 
# - **Cantidad de Reviews**: 479,343  
# - **Cantidad de Comentarios**: 233,588  
# - **Cantidad de Lugares**: 1,390  
# - **Cantidad de Usuarios Únicos**: 205,836  

# ## 2. Ingeniería de Datos:
# ### 2.1 Asignación de formato datetime

# Convertir la columna de fecha a tipo datetime manejando errores y formato específico
df_base['fecha'] = pd.to_datetime(df_base['fecha'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')

# Crear nuevas columnas
df_base['year'] = df_base['fecha'].dt.year
df_base['month'] = df_base['fecha'].dt.month
df_base['day'] = df_base['fecha'].dt.day
df_base['time'] = df_base['fecha'].dt.time

# Verificar las primeras filas
print(df_base[['fecha', 'year', 'month', 'day', 'time']].head())


# Se obsevan valores Nulos por lo que se deben trabajar

# Cantidad de valores nulos (NaN o NaT) en la columna 'fecha'
nan_count = df_base['fecha'].isna().sum()

# Cantidad de valores no nulos (válidos) en la columna 'fecha'
not_nan_count = df_base['fecha'].notna().sum()

# Imprimir resultados
print(f"Cantidad de valores NaN/NaT en 'fecha': {nan_count}")
print(f"Cantidad de valores válidos en 'fecha': {not_nan_count}")


# Filtrar filas donde 'fecha' es NaN/NaT
nan_fecha = df_base[df_base['fecha'].isna()]

# Contar cuántas de esas filas tienen datos en 'review'
datos_en_review = nan_fecha['review'].notna().sum()

# Imprimir resultados
print(f"Cantidad de valores NaN/NaT en 'fecha': {len(nan_fecha)}")
print(f"De ellos, cantidad de datos en 'review': {datos_en_review}")


# Dado que una gran parte de los valores nulos tienen "review" se decide rellenar los NaN o NaT buscando cada id_usuario y otras reviews que haya realizado.

# Crear una función para rellenar valores NaN/NaT basados en id_usuario
def rellenar_fecha_por_usuario(df, columnas_fecha, id_col):
    for col in columnas_fecha:
        # Rellenar valores NaN basándose en el promedio de la columna por id_usuario
        df[col] = df.groupby(id_col)[col].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    return df

# Columnas de fecha que necesitan ser rellenadas
columnas_fecha = ['year', 'month', 'day']

# Aplicar la función al DataFrame
df_base = rellenar_fecha_por_usuario(df_base, columnas_fecha, 'id_usuario')

# Verificar el resultado
print(df_base[['id_usuario', 'year', 'month', 'day']].head())


# Contar valores NaN en cada columna relacionada con la fecha
nan_count = df_base[['year', 'month', 'day']].isna().sum()

# Imprimir resultados
print("Cantidad de valores NaN restantes por columna:")
print(nan_count)


# Los valores aún faltantes, se rellenan utilizando la mediana para cada variable.

# Función para rellenar NaN con la mediana por place_id
def rellenar_fecha_por_place_id(df, columnas_fecha, place_col):
    for col in columnas_fecha:
        # Rellenar valores NaN con la mediana del grupo por place_id
        df[col] = df.groupby(place_col)[col].transform(lambda x: x.fillna(x.median()))
    return df

# Columnas de fecha que necesitan ser rellenadas
columnas_fecha = ['year', 'month', 'day']

# Aplicar la función al DataFrame
df_base = rellenar_fecha_por_place_id(df_base, columnas_fecha, 'place_id')

# Verificar el resultado
print("Cantidad de valores NaN restantes por columna después del rellenado:")
print(df_base[columnas_fecha].isna().sum())


# Verificar si quedan NaN y reemplazarlos con un valor predeterminado (opcional)

df_base.fillna({'year': 0, 'month': 1, 'day': 1}, inplace=True)


# Convertir las columnas de fecha a tipo entero
df_base[['year', 'month', 'day']] = df_base[['year', 'month', 'day']].astype(int)


# Verificar el resultado
print(df_base[['year', 'month', 'day']].dtypes)
df_base[['year', 'month', 'day']].head()


# Se decide rellenar los NaN o NaT buscando cada id_usuario y otras reviews que haya realizado.

df_base.head()


# Se visualizan las columnas para ver cual de ellas se eliminaran.

df_base.columns


# Se decide eliminar:
# - user_name, pues es redundante al id_usuario
# - day y time, ya que no son necesarios para el análisis a realizar
# - link_review, es innecesario para el análisis
# - id: redundante al indice
# - id_review: No utilizado

# 3. Eliminar columnas innecesarias: 'user_name', 'day', 'time', 'link_review', 'id', 'id_review'
df_base = df_base.drop(columns=['user_name', 'day', 'time', 'link_review', 'Id', 'fecha','id_review'])

# 4. Renombrar la columna 'name' a 'place_name'
df_base = df_base.rename(columns={'name': 'place_name'})

# 5. Incorporar una columna 'rating' que sea el promedio de 'stars' por 'place_id'
df_base['rating'] = df_base.groupby('place_id')['stars'].transform('mean')

# 6. Resetear el índice para que sea un número consecutivo
df_base = df_base.reset_index(drop=True)

# Verificar el resultado
df_base.head()


# Guardar el DataFrame en un archivo Excel
# df_base.to_excel("./data/reviews_Los_Rios.xlsx", index=False)

print("El nuevo dataset se ha guardado como 'reviews_Los_Rios.xlsx'.")


list_placeids = df_base['place_id'].unique()
print(len(list_placeids))
print(len(df_base['id_usuario'].unique()))


# ## 3. Enriquecimiento de los datos
# 

import requests
import pandas as pd

API_KEY = "xx" 
list_placeids = df_base['place_id'].unique()  # Lista única de place_id desde el DataFrame


# 
# ### 3.1 Geocoding
# 

# #### Buscar las coordenadas de un place_id con la API de Google Maps: Geocoding

# Función para obtener geocodificación por place_id
def get_geocode(api_key, place_id):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"place_id": place_id, "key": api_key}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            result = data["results"][0]
            latitude = result["geometry"]["location"]["lat"]
            longitude = result["geometry"]["location"]["lng"]
            return {
                "latitude": latitude,
                "longitude": longitude
            }
        else:
            return {"error": "No se encontraron resultados"}
    else:
        return {"error": f"Error en la solicitud: {response.status_code}"}

# Lista de place_ids únicos desde el DataFrame
list_placeids = df_base['place_id'].unique()

# Agregar columnas para coordenadas al DataFrame
df_base['Latitud'] = None
df_base['Longitud'] = None

# Llenar coordenadas por place_id
for place_id in list_placeids:
    resultado = get_geocode(API_KEY, place_id)
    if "error" in resultado:
        print(f"Error para {place_id}: {resultado['error']}")
    else:
        df_base.loc[df_base['place_id'] == place_id, 'Latitud'] = resultado["latitude"]
        df_base.loc[df_base['place_id'] == place_id, 'Longitud'] = resultado["longitude"]


# ### 3.2 Places

# Función para obtener detalles de un lugar por place_id
def get_place_details(api_key, place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "key": api_key}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "result" in data:
            result = data["result"]
            types = result.get("types", [])  # Tipos/categorías del lugar
            address_components = result.get("address_components", [])
            city = None
            
            # Buscar la ciudad entre los componentes de dirección
            for component in address_components:
                if "locality" in component["types"]:
                    city = component["long_name"]
                    break
            
            return {
                "types": types,
                "city": city
            }
        else:
            return {"error": "No se encontraron resultados"}
    else:
        return {"error": f"Error en la solicitud: {response.status_code}"}

# Lista de place_ids únicos
list_placeids = df_base['place_id'].unique()

# Agregar columnas para tipo, subtipos y ciudad
df_base['types'] = None
df_base['city'] = None

# Obtener información para cada place_id
for place_id in list_placeids:
    resultado = get_place_details(API_KEY, place_id)
    if "error" in resultado:
        print(f"Error para {place_id}: {resultado['error']}")
    else:
        types = ", ".join(resultado["types"]) if resultado["types"] else None
        city = resultado["city"]
        df_base.loc[df_base['place_id'] == place_id, 'types'] = types
        df_base.loc[df_base['place_id'] == place_id, 'city'] = city


df_base.head()


# Ver los valores únicos en la columna 'types'
unique_types = df_base['types'].unique()

# Mostrar los resultados
print("Valores únicos en 'types':")
print(unique_types)


# Dividir la columna 'types' en varias partes separadas por comas
df_base[['category', 'type', 'subtype']] = df_base['types'].str.split(',', n=2, expand=True)

# Eliminar espacios adicionales si los hay
df_base['category'] = df_base['category'].str.strip()
df_base['type'] = df_base['type'].str.strip()
df_base['subtype'] = df_base['subtype'].str.strip()

# Verificar el resultado
print(df_base[['types', 'category', 'type', 'subtype']].head())

df_base.head()


# ## 4. Análisis de patrones

# Agrupar por 'year' y contar la cantidad de registros (reviews) por año
reviews_per_year = df_base.groupby('year').size().reset_index(name='review')

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.bar(reviews_per_year['year'], reviews_per_year['review'], label='cantidad')

# Personalización de la gráfica
plt.title('Cantidad de Reviews por Año')
plt.xlabel('Year')
plt.ylabel('Cantidad')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xticks(reviews_per_year['year'], rotation=45)
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# Poner % en las barras!!


# Visualizamos la cantidad de reviews existentes por cada categoría, donde el ranking top 10 es el siguiente:
# 
#     1.Restaurant
#     2. Tourist_attraction
#     3. Point_of_interest
#     4. Lodging
#     5. Park
#     6. Natural_feautre
#     7. Bar
#     8. Museum
#     9. Food
#     10. Shopping_mall

# Categorías a analizar
categorias_filtradas = ['restaurant', 'tourist_attraction', 'point_of_interest', 'lodging', 'park', 'natural_feature']

# Crear un diccionario para almacenar el Top 10 por cada categoría
top_10_por_categoria = {}

for categoria in categorias_filtradas:
    # Filtrar el DataFrame por la categoría actual
    df_categoria = df_base[df_base['category'] == categoria]
    
    # Contar la cantidad de reviews por place_name
    top_places = df_categoria.groupby('place_name').size().reset_index(name='cantidad')
    
    # Ordenar por cantidad de reviews en orden descendente y seleccionar el Top 10
    top_10 = top_places.sort_values(by='cantidad', ascending=False).head(10)
    
    # Guardar el resultado en el diccionario
    top_10_por_categoria[categoria] = top_10

    # Mostrar el Top 10 de la categoría actual
    print(f"\nTop 10 lugares para la categoría '{categoria}':")
    print(top_10)

# Visualizar el resultado en gráficos separados
import matplotlib.pyplot as plt

for categoria, top_10 in top_10_por_categoria.items():
    plt.figure(figsize=(12, 6))
    plt.barh(top_10['place_name'], top_10['cantidad'], color='lightblue')
    plt.xlabel('Cantidad de Reviews', fontsize=12)
    plt.ylabel('Place Name', fontsize=12)
    plt.title(f"Top 10 Places en la Categoría '{categoria}'", fontsize=14)
    plt.gca().invert_yaxis()  # Lugar con más reviews en la parte superior
    plt.tight_layout()
    plt.show()


# Agrupar por 'category' y contar la cantidad de registros (reviews) por categoría
reviews_per_category = df_base.groupby('category').size().reset_index(name='cantidad')

# Ordenar por cantidad para mejor visualización
reviews_per_category = reviews_per_category.sort_values(by='cantidad', ascending=False)

# Crear la gráfica de barras
plt.figure(figsize=(12, 6))
plt.bar(reviews_per_category['category'], reviews_per_category['cantidad'], color='skyblue')

# Personalización de la gráfica
plt.title('Cantidad de Reviews por Categoría', fontsize=14)
plt.xlabel('Categoría', fontsize=12)
plt.ylabel('Cantidad de Reviews', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()


# Del top 6, se busca analizar el top 10 de lugares con más reviews:

# Filtrar el DataFrame para las categorías específicas
categorias_filtradas = ['restaurant', 'tourist_attraction', 'point_of_interest', 'lodging', 'park', 'natural_feature']
df_filtrado = df_base[df_base['category'].isin(categorias_filtradas)]

# Contar la cantidad de reviews por place_name
top_places = df_filtrado.groupby('place_name').size().reset_index(name='cantidad')

# Ordenar por cantidad de reviews en orden descendente
top_places = top_places.sort_values(by='cantidad', ascending=False)

# Seleccionar el Top 10
top_10_places = top_places.head(10)

# Mostrar el Top 10
print("Top 10 lugares por cantidad de reviews en las categorías seleccionadas:")
print(top_10_places)

# Visualización opcional en un gráfico
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.barh(top_10_places['place_name'], top_10_places['cantidad'], color='lightblue')
plt.xlabel('Cantidad de Reviews', fontsize=12)
plt.ylabel('Place Name', fontsize=12)
plt.title('Top 10 Places por Categorías Seleccionadas', fontsize=14)
plt.gca().invert_yaxis()  # Invertir el eje Y para que el lugar con más reviews esté arriba
plt.tight_layout()
plt.show()


# Para poder trabajar la red, filtramos por los usuarios que han visitado más de 1 lugar:

# Contar cuántos lugares únicos ha visitado cada usuario
lugares_por_usuario = df_base.groupby('id_usuario')['place_name'].nunique()

# Calcular usuarios que han visitado más de 1 lugar
usuarios_mas_de_1_lugar = (lugares_por_usuario > 1).sum()

# Calcular usuarios que han visitado exactamente 1 lugar
usuarios_solo_1_lugar = (lugares_por_usuario == 1).sum()

# Mostrar los resultados
print(f"Usuarios que han visitado más de 1 lugar: {usuarios_mas_de_1_lugar}")
print(f"Usuarios que han visitado solo 1 lugar: {usuarios_solo_1_lugar}")


# Verificar si hay usuarios con 'place_name' nulo
usuarios_sin_visitas = df_base[df_base['place_name'].isna()]['id_usuario'].nunique()

# Mostrar el resultado
print(f"Cantidad de usuarios que no han visitado ningún lugar: {usuarios_sin_visitas}")


# Contar cuántos lugares únicos ha visitado cada usuario
lugares_por_usuario = df_base.groupby('id_usuario')['place_name'].nunique()

# Identificar los usuarios que han visitado más de 1 lugar
usuarios_mas_de_1 = lugares_por_usuario[lugares_por_usuario > 1].index

# Crear un nuevo DataFrame con los usuarios que han visitado más de 1 lugar
nuevo_df = df_base[df_base['id_usuario'].isin(usuarios_mas_de_1)].copy()

# Verificar el resultado
print(f"Cantidad de usuarios en el nuevo dataset: {nuevo_df['id_usuario'].nunique()}")
print(f"Tamaño del nuevo dataset: {nuevo_df.shape}")


# Guardar el nuevo dataset en Excel
# nuevo_df.to_excel("Redes_coocurrencia.xlsx", index=False)

print("El nuevo dataset se ha guardado como 'Redes_coocurrencia.xlsx'.")


df_base = pd.read_excel("./Redes_coocurrencia.xlsx").reset_index()
df_base = df_base.rename({'index':'Id'}, axis='columns')


df_base


# #### Resumen Estadístico General

# Resumen estadístico general para variables numéricas
print("Resumen estadístico general:")
print(df_base.describe())

# Resumen para variables categóricas (frecuencia de valores únicos)
print("\nFrecuencia de valores únicos en variables categóricas:")
print(df_base.nunique())


# Cantidad de usuarios únicos
cantidad_usuarios = df_base['id_usuario'].nunique()
print(f"Cantidad de usuarios únicos: {cantidad_usuarios}")

# Cantidad de destinos únicos
cantidad_destinos = df_base['place_name'].nunique()
print(f"Cantidad de destinos únicos: {cantidad_destinos}")


# Los nodos principales de la red corresponden a los **usuarios** y a los **destinos**. En total contamos con :
# 
#     Cantidad de usuarios únicos: 82983
#     Cantidad de destinos únicos: 1373

# #### Distribución de reviews por Usuario y Lugar

# Reviews por usuario
reviews_por_usuario = df_base.groupby('id_usuario').size()
print("\nDistribución de reviews por usuario:")
print(reviews_por_usuario.describe())

# Reviews por destino
reviews_por_destino = df_base.groupby('place_name').size()
print("\nDistribución de reviews por destino:")
print(reviews_por_destino.describe())


# **Distribución de reviews por Usuario**:
# 
# `Count:` Hay un total de 82.983 usuarios en el dataset que han realizado reviews.
# 
# `Mean:` En promedio, cada usuario ha realizado 4.30 reviews.
# 
# `Std:` La desviación estándar es 4.48, lo que indica una dispersión significativa en la cantidad de reviews realizadas por los usuarios.
# 
# `Min:` El usuario con menos actividad realizó 1 review.
# 
# `Percentil 25%:` El 25% de los usuarios realizaron 2 reviews o menos.
# 
# `Percentil 50%`: El 50% de los usuarios realizaron 3 reviews o menos.
# 
# `Percentil 75%`: El 75% de los usuarios realizaron 5 reviews o menos.
# 
# `Max:` El usuario más activo realizó 146 reviews.
# 
# Podemos ver que la mayoría de los usuarios realiza pocas reviews, pero hay algunos usuarios que destacan por ser mucho más activos (outliers).
# 
# **Distribución de reviews por Destino**:
# 
# `Count:` Hay 1,373 destinos turísticos únicos en el dataset.
# 
# `Mean:` En promedio, cada destino ha recibido 259.63 reviews.
# 
# `Std:` La desviación estándar es 826.83, lo que indica que algunos destinos reciben muchas más reviews que otros.
# 
# `Min:` El destino menos visitado tiene 1 review.
# 
# `Percentil 25%:` El 25% de los destinos han recibido 18 reviews o menos.
# 
# `Percentil 50%:` El 50% de los destinos han recibido 56 reviews o menos.
# 
# `Percentil 75%:` El 75% de los destinos han recibido 178 reviews o menos.
# 
# `Max:` El destino más popular ha recibido 13,253 reviews.
# 
# Podemos ver que la distribución de reviews por destino es altamente desigual: algunos destinos son extremadamente populares (outliers), mientras que muchos reciben pocas visitas o calificaciones.

# Top 10 destinos más visitados con porcentaje
top_destinos = df_base['place_name'].value_counts().head(10)
total_visitas = df_base['place_name'].value_counts().sum()

# Agregar porcentaje al Top 10
top_destinos_porcentaje = top_destinos.apply(lambda x: (x / total_visitas) * 100)

# Mostrar resultados
print("\nTop 10 destinos más visitados con porcentaje del total:")
top_destinos_con_porcentaje = pd.DataFrame({
    'Destino': top_destinos.index,
    'Visitas': top_destinos.values,
    'Porcentaje (%)': top_destinos_porcentaje.values
})
print(top_destinos_con_porcentaje)

# Top 10 usuarios con más reviews
top_usuarios = df_base['id_usuario'].value_counts().head(10)
total_reviews = df_base['id_usuario'].value_counts().sum()

# Agregar porcentaje al Top 10
top_usuarios_porcentaje = top_usuarios.apply(lambda x: (x / total_reviews) * 100)

# Mostrar resultados
print("\nTop 10 usuarios con más reviews con porcentaje del total:")
top_usuarios_con_porcentaje = pd.DataFrame({
    'Usuario': top_usuarios.index,
    'Reviews': top_usuarios.values,
    'Porcentaje (%)': top_usuarios_porcentaje.values
})
print(top_usuarios_con_porcentaje)




plt.figure(figsize=(10, 6))
plt.barh(top_destinos_con_porcentaje['Destino'], top_destinos_con_porcentaje['Visitas'], color='skyblue', edgecolor='black')
plt.xlabel('Número de Visitas')
plt.ylabel('Destinos')
plt.title('Top 10 Destinos Más Visitados')
plt.gca().invert_yaxis()  # Colocar el destino más popular arriba
plt.tight_layout()

# Mostrar porcentajes en las barras
for i, (visitas, porcentaje) in enumerate(zip(top_destinos_con_porcentaje['Visitas'], top_destinos_con_porcentaje['Porcentaje (%)'])):
    plt.text(visitas + 50, i, f"{porcentaje:.2f}%", va='center')

plt.show()



# Gráfico de barras horizontales para usuarios con más reviews
plt.figure(figsize=(10, 6))
plt.barh(top_usuarios_con_porcentaje['Usuario'], top_usuarios_con_porcentaje['Reviews'], color='lightcoral', edgecolor='black')
plt.xlabel('Número de Reviews')
plt.ylabel('Usuarios')
plt.title('Top 10 Usuarios Más Activos')
plt.gca().invert_yaxis()  # Colocar el usuario más activo arriba
plt.tight_layout()

# Mostrar porcentajes en las barras
for i, (reviews, porcentaje) in enumerate(zip(top_usuarios_con_porcentaje['Reviews'], top_usuarios_con_porcentaje['Porcentaje (%)'])):
    plt.text(reviews + 2, i, f"{porcentaje:.2f}%", va='center')

plt.show()



# #### Comportamiento de las calificaciones (stars)

# Resumen de la variable 'stars'
print("\nResumen de calificaciones (stars):")
print(df_base['stars'].describe())

# Distribución de calificaciones
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
df_base['stars'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title("Distribución de Calificaciones (Stars)")
plt.xlabel("Calificación (Stars)")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.show()


# `Total de calificaciones:` Hay 356,475 reviews en total en el dataset.
# 
# `Calificación promedio:` La calificación promedio es 4.49, lo que sugiere que la mayoría de las evaluaciones son positivas.
# 
# `Desviación estándar (std):` La desviación estándar es 0.92, lo que indica que la mayoría de las calificaciones están cercanas al promedio, aunque hay algo de variabilidad.
# 
# `Mediana:` La calificación mediana es 5.0, lo que significa que al menos el 50% de las reviews tienen la calificación máxima.
# 
# `Mínimo:` La calificación más baja es 1.0, lo que indica que también hay destinos con evaluaciones muy negativas.
# 
# `Máximo:` La calificación más alta es 5.0, lo cual es esperado ya que las calificaciones están en una escala de 1 a 5.
# 
# La barra más alta corresponde a la calificación de 5.0, que tiene una frecuencia superior a 250,000 reviews, es la calificación más común en el dataset. Las calificaciones de 1.0, 2.0 y 3.0 tienen frecuencias mucho más bajas, por lo que podemos deducir que las reseñas negativas son menos comunes.
# 
# Con respecto a las evaluaciones positivas, podemos creer que pudiera haber un sesgo en los datos, donde solo usuarios satisfechos evaluan los lugares. Y en este sentido, creemos importante analizar en detalle los lugares con evaluaciones bajas que pudieran estar enfrentando problemas con la experiencia de los usuarios.
# 
# Al tener un alto número de evaluaciones positivas, también consideramos importante considerar métricas adicionales como la frecuencia de visitas y el tipo de comentarios que realizanm para poder afrontar un correcto sistema de recomendación.

# #### Frecuencia de visitas para identificar los destinos más visitados y los usuarios más activos.

# Top 10 destinos más visitados con porcentaje
top_destinos = df_base['place_name'].value_counts().head(10)
total_visitas = df_base['place_name'].value_counts().sum()

# Agregar porcentaje al Top 10
top_destinos_porcentaje = top_destinos.apply(lambda x: (x / total_visitas) * 100)

# Mostrar resultados
print("\nTop 10 destinos más visitados con porcentaje del total:")
top_destinos_con_porcentaje = pd.DataFrame({
    'Destino': top_destinos.index,
    'Visitas': top_destinos.values,
    'Porcentaje (%)': top_destinos_porcentaje.values
})
print(top_destinos_con_porcentaje)

# Top 10 usuarios con más reviews
top_usuarios = df_base['id_usuario'].value_counts().head(10)
total_reviews = df_base['id_usuario'].value_counts().sum()

# Agregar porcentaje al Top 10
top_usuarios_porcentaje = top_usuarios.apply(lambda x: (x / total_reviews) * 100)

# Mostrar resultados
print("\nTop 10 usuarios con más reviews con porcentaje del total:")
top_usuarios_con_porcentaje = pd.DataFrame({
    'Usuario': top_usuarios.index,
    'Reviews': top_usuarios.values,
    'Porcentaje (%)': top_usuarios_porcentaje.values
})
print(top_usuarios_con_porcentaje)




# #### ¿Qué lugar visitan los usuarios más activos?

# Analizamos cuáles son los destinos más frecuentados por el top 10 de usuarios que tienen la mayor cantidad de interacciones (reviews).

# Top 10 usuarios más activos
top_usuarios = df_base['id_usuario'].value_counts().head(10).index

# Filtrar los destinos visitados por los usuarios más activos
destinos_usuarios_activos = df_base[df_base['id_usuario'].isin(top_usuarios)]

# Contar las visitas de cada usuario a los destinos
destinos_por_usuario = destinos_usuarios_activos.groupby(['id_usuario', 'place_name']).size().reset_index(name='visitas')

# Mostrar los destinos más visitados por los usuarios más activos
print("Destinos visitados por los usuarios más activos:")
print(destinos_por_usuario.sort_values(by=['id_usuario', 'visitas'], ascending=[True, False]))


# Obtener destinos únicos visitados por los usuarios más activos
destinos_unicos = destinos_usuarios_activos['place_name'].unique()

# Mostrar los destinos únicos
print(f"Cantidad de destinos únicos visitados por los usuarios más activos: {len(destinos_unicos)}")
print("\nDestinos únicos visitados por los usuarios más activos:")
print(destinos_unicos)


# Contar cuántos destinos de cada categoría han visitado los usuarios más activos
categorias_por_usuario = destinos_usuarios_activos.groupby(['id_usuario', 'category']).size().reset_index(name='conteo')
print("Categorías más visitadas por los usuarios más activos:")
print(categorias_por_usuario.sort_values(by=['id_usuario', 'conteo'], ascending=[True, False]))


# La categoria `Restaurant` domina como categoría más visitada. 
# 
# - El usuario más activo visitó 28 destinos en la categoría "restaurant", seguido por "bar" (10 destinos) y "park" (8 destinos).
# 
# - También hay visitas a categorías menos comunes como "cafe", "store" y "casino".
# 
# **Las categorías populares están relacionadas con actividades recreativas y gastronómicas, lo que sugiere que los usuarios más activos buscan experiencias sociales y culturales.**

# ### TOP 3 Categorias.

# #### Analizamos la categoria `Restaurant`:

# Filtrar datos para la categoría "restaurant"
calificaciones_restaurant = df_base[df_base['category'] == 'restaurant']

# Top 10 destinos con las evaluaciones más altas (≥4)
top_10_altas = calificaciones_restaurant[calificaciones_restaurant['stars'] >= 4] \
    .groupby('place_name')['stars'].mean() \
    .reset_index(name='calificacion_promedio') \
    .sort_values(by='calificacion_promedio', ascending=False) \
    .head(10)

# Top 10 destinos con las evaluaciones más bajas (≤2)
top_10_bajas = calificaciones_restaurant[calificaciones_restaurant['stars'] <= 2] \
    .groupby('place_name')['stars'].mean() \
    .reset_index(name='calificacion_promedio') \
    .sort_values(by='calificacion_promedio', ascending=True) \
    .head(10)

# Mostrar los resultados
print("Top 10 destinos con las evaluaciones más altas en la categoría 'restaurant':")
print(top_10_altas)

print("\nTop 10 destinos con las evaluaciones más bajas en la categoría 'restaurant':")
print(top_10_bajas)


# Gráfico de barras para las evaluaciones más altas

plt.figure(figsize=(10, 6))
plt.barh(top_10_altas['place_name'], top_10_altas['calificacion_promedio'], color='lightgreen', edgecolor='black')
plt.title("Top 10 Evaluaciones Más Altas en 'Restaurant'")
plt.xlabel("Calificación Promedio")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Gráfico de barras para las evaluaciones más bajas
plt.figure(figsize=(10, 6))
plt.barh(top_10_bajas['place_name'], top_10_bajas['calificacion_promedio'], color='lightcoral', edgecolor='black')
plt.title("Top 10 Evaluaciones Más Bajas en 'Restaurant'")
plt.xlabel("Calificación Promedio")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# #### Analizamos la categoria `Bar`:

# Filtrar datos para la categoría "bar"
calificaciones_bar = df_base[df_base['category'] == 'bar']

# Top 10 destinos con las evaluaciones más altas (≥4)
top_10_altas = calificaciones_bar[calificaciones_bar['stars'] >= 4] \
    .groupby('place_name')['stars'].mean() \
    .reset_index(name='calificacion_promedio') \
    .sort_values(by='calificacion_promedio', ascending=False) \
    .head(10)

# Top 10 destinos con las evaluaciones más bajas (≤2)
top_10_bajas = calificaciones_bar[calificaciones_bar['stars'] <= 2] \
    .groupby('place_name')['stars'].mean() \
    .reset_index(name='calificacion_promedio') \
    .sort_values(by='calificacion_promedio', ascending=True) \
    .head(10)

# Mostrar los resultados
print("Top 10 destinos con las evaluaciones más altas en la categoría 'bar':")
print(top_10_altas)

print("\nTop 10 destinos con las evaluaciones más bajas en la categoría 'bar':")
print(top_10_bajas)


# Gráfico de barras para las evaluaciones más altas

plt.figure(figsize=(10, 6))
plt.barh(top_10_altas['place_name'], top_10_altas['calificacion_promedio'], color='lightgreen', edgecolor='black')
plt.title("Top 10 Evaluaciones Más Altas en 'Bar'")
plt.xlabel("Calificación Promedio")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Gráfico de barras para las evaluaciones más bajas
plt.figure(figsize=(10, 6))
plt.barh(top_10_bajas['place_name'], top_10_bajas['calificacion_promedio'], color='lightcoral', edgecolor='black')
plt.title("Top 10 Evaluaciones Más Bajas en 'Bar'")
plt.xlabel("Calificación Promedio")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# #### Analizamos la categoria `Park`:

# Filtrar datos para la categoría "park"
calificaciones_park = df_base[df_base['category'] == 'park']

# Top 10 destinos con las evaluaciones más altas (≥4)
top_10_altas = calificaciones_park[calificaciones_park['stars'] >= 4] \
    .groupby('place_name')['stars'].mean() \
    .reset_index(name='calificacion_promedio') \
    .sort_values(by='calificacion_promedio', ascending=False) \
    .head(10)

# Top 10 destinos con las evaluaciones más bajas (≤2)
top_10_bajas = calificaciones_bar[calificaciones_bar['stars'] <= 2] \
    .groupby('place_name')['stars'].mean() \
    .reset_index(name='calificacion_promedio') \
    .sort_values(by='calificacion_promedio', ascending=True) \
    .head(10)

# Mostrar los resultados
print("Top 10 destinos con las evaluaciones más altas en la categoría 'park':")
print(top_10_altas)

print("\nTop 10 destinos con las evaluaciones más bajas en la categoría 'park':")
print(top_10_bajas)


# Gráfico de barras para las evaluaciones más altas

plt.figure(figsize=(10, 6))
plt.barh(top_10_altas['place_name'], top_10_altas['calificacion_promedio'], color='lightgreen', edgecolor='black')
plt.title("Top 10 Evaluaciones Más Altas en 'park'")
plt.xlabel("Calificación Promedio")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Gráfico de barras para las evaluaciones más bajas
plt.figure(figsize=(10, 6))
plt.barh(top_10_bajas['place_name'], top_10_bajas['calificacion_promedio'], color='lightcoral', edgecolor='black')
plt.title("Top 10 Evaluaciones Más Bajas en 'park'")
plt.xlabel("Calificación Promedio")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# De estos graficos y los anteriormente presentados se puede observar que hay una baja coocurrencia de los lugares con mayor cantidad de reviews y los con mejor calificación, esto implica que las notas de estos lugares se encuentran infladas en base a la baja cantidad de reviews que tienen.

# #### ¿Qué ciudad visitan los usuarios más activos?

ciudades_por_usuario = destinos_usuarios_activos.groupby(['id_usuario', 'city']).size().reset_index(name='conteo')
print("Ciudades más visitadas por los usuarios más activos:")
ciudades_por_usuario.sort_values(by=['id_usuario', 'conteo'], ascending=[True, False])


# La ciudad `Valdivia` es la ciudad más visitada.
# 
# - El usuario más activo realizó 61 visitas a destinos en Valdivia, lo que la convierte en un punto de interés clave.
# 
# - Otras ciudades visitadas por los usuarios más activos incluyen "San José de Mariquina", "Corral" y "Panguipulli", pero con un número significativamente menor de visitas (1-2 destinos).

# Calcular el promedio de calificaciones (stars) de los usuarios más activos
calificaciones_por_usuario = destinos_usuarios_activos.groupby('id_usuario')['stars'].mean().reset_index(name='calificacion_promedio')

# Comparar con la calificación promedio general
calificacion_promedio_general = df_base['stars'].mean()

print(f"Calificación promedio general: {calificacion_promedio_general:.2f}")
print("\nCalificaciones promedio de los usuarios más activos:")
print(calificaciones_por_usuario)

# Visualización: comparar calificaciones promedio
plt.figure(figsize=(10, 6))
plt.bar(calificaciones_por_usuario['id_usuario'].astype(str), calificaciones_por_usuario['calificacion_promedio'], color='skyblue', edgecolor='black')
plt.axhline(calificacion_promedio_general, color='red', linestyle='--', label='Promedio general')
plt.xlabel("Usuarios")
plt.ylabel("Calificación Promedio")
plt.title("Calificaciones Promedio de los Usuarios Más Activos")
plt.xticks(rotation=90)
plt.legend()
plt.show()


# Las calificaciones promedio de los usuarios más activos varían entre 4.378 y 4.869, lo que indica diferencias leves. La mayoría de las calificaciones promedio están por encima del promedio general (representado por la línea roja en el gráfico), lo que sugiere que los usuarios más activos tienden a dar calificaciones más altas.
# 
# Es posible observar un sesgo positivo entre los usuarios más frecuentes, o también podría ser que los destinos frecuentados por estos usuarios generalmente reciben buenas calificaciones.
# 
# Se identifica un usuario cuya calificación promedio (4.378) está ligeramente por debajo de las demás, aunque sigue siendo positiva. **Este usuario podría ser considerado más crítico en comparación con los otros usuarios más activos y podría ser útil para futuros análisis.** `Usuario 17105786385826823095` y `Usuario 107285632152162720`.

# Se Analiza las calificaciones bajas (≤2) y altas (≥4) realizadas por los usuarios más activos, y se identifica las siguientes observaciones clave:

# Contar calificaciones bajas (1-2 estrellas) y altas (4-5 estrellas) por usuario
calificaciones_extremas = destinos_usuarios_activos.groupby('id_usuario')['stars'].apply(
    lambda x: pd.Series({
        'calificaciones_bajas': sum(x <= 2),
        'calificaciones_altas': sum(x >= 4)
    })
).reset_index()

print("Calificaciones bajas y altas por usuario:")
print(calificaciones_extremas)

# Visualizar proporción de calificaciones extremas
ax = calificaciones_extremas.plot(
    x='id_usuario',
    kind='bar',
    stacked=True,
    figsize=(12, 7),
    color=['salmon', 'skyblue'],
    edgecolor='black'
)
for i, patch in enumerate(ax.patches):
    if i % 2 == 0:
        patch.set_facecolor('salmon')  # Set the color for Column 1
    else:
        patch.set_facecolor('skyblue') 

# Añadir símbolos y leyenda con explicación
for bar, label, symbol in zip(ax.patches, ['Bajas', 'Altas'], ['↓', '↑']):
    height = bar.get_height()
    if height > 0:  # Añadir solo para barras visibles
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,  # Centro de la barra
            f"{symbol}",
            ha='center',
            va='center',
            fontsize=10,
            color='white'
        )

# Personalizar el gráfico
plt.title("Calificaciones Extremas por Usuario")
plt.ylabel("Cantidad de Calificaciones")
plt.xlabel("Usuarios")
plt.legend(
    ["Calificaciones Bajas (<= 2)", "Calificaciones Altas (>= 4)"],
    loc='upper right'
)
plt.tight_layout()
plt.show()


# `Usuarios con calificaciones bajas (≤2):`
# La mayoría de los usuarios tienen muy pocas calificaciones bajas (entre 0 y 4). Esto indica que estos usuarios suelen otorgar calificaciones positivas o moderadas en lugar de críticas negativas.
# Por ejemplo, algunos usuarios como 1024646466013135463 y 107285632152162720 no dieron ninguna calificación baja, mientras que otros, como 117105786385826823095, dieron hasta 12 calificaciones bajas.

# `Usuarios con calificaciones altas (≥4):`Todos los usuarios tienen un número considerablemente mayor de calificaciones altas en comparación con calificaciones bajas. Algunos usuarios, como 117105786385826823095, destacan por haber otorgado hasta 130 calificaciones altas, lo que refleja un comportamiento altamente positivo.

# #### Relación entre calificaciones bajas y categorías. Aca se consideran todos los usuarios, no solo el top 10 que mas calificaciones ha realizado.

# Filtrar las calificaciones bajas (≤2)
calificaciones_bajas = df_base[df_base['stars'] <= 2]

# Contar cuántas calificaciones bajas tiene cada categoría
calificaciones_bajas_por_categoria = calificaciones_bajas.groupby('category').size().reset_index(name='conteo')

# Ordenar por el número de calificaciones bajas
calificaciones_bajas_por_categoria = calificaciones_bajas_por_categoria.sort_values(by='conteo', ascending=False)

# Mostrar los resultados
print("Calificaciones bajas por categoría:")
print(calificaciones_bajas_por_categoria)

# Visualizar
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(calificaciones_bajas_por_categoria['category'], calificaciones_bajas_por_categoria['conteo'], color='salmon', edgecolor='black')
plt.title("Calificaciones Bajas por Categoría")
plt.xlabel("Categoría")
plt.ylabel("Cantidad de Calificaciones Bajas")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Contar calificaciones bajas por destino
calificaciones_bajas_por_destino = calificaciones_bajas.groupby('place_name').size().reset_index(name='conteo')

# Ordenar por número de calificaciones bajas
calificaciones_bajas_por_destino = calificaciones_bajas_por_destino.sort_values(by='conteo', ascending=False).head(10)

# Mostrar los resultados
print("Destinos con más calificaciones bajas:")
print(calificaciones_bajas_por_destino)

# Visualizar
plt.figure(figsize=(10, 6))
plt.barh(calificaciones_bajas_por_destino['place_name'], calificaciones_bajas_por_destino['conteo'], color='lightcoral', edgecolor='black')
plt.title("Top 10 Destinos con Calificaciones Bajas")
plt.xlabel("Cantidad de Calificaciones Bajas")
plt.ylabel("Destinos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# **Es probable que la categoría más visitada, "restaurant", esté asociada principalmente con destinos en ciudades como Valdivia, que concentra una gran cantidad de actividades turísticas y gastronómicas.**

# #### ¿Los destinos más visitados tienen mejores calificaciones promedio?

# Relación entre calificaciones promedio y visitas por destino
df_calif_visitas = df_base.groupby('place_name').agg(
    calificacion_promedio=('stars', 'mean'),
    cantidad_visitas=('stars', 'count')
).reset_index()

# Visualización
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df_calif_visitas['calificacion_promedio'], df_calif_visitas['cantidad_visitas'], alpha=0.6, color='blue')
plt.title("Relación entre Número de Visitas y Calificación Promedio")
plt.ylabel("Cantidad de Visitas")
plt.xlabel("Calificación Promedio")
plt.grid(alpha=0.3)
plt.show()


# La mayoría de los destinos tienen un número bajo de visitas grupados en el rango de 0 a 1000. Estos destinos suelen tener calificaciones promedio distribuidas entre 3.0 y 5.0, con una tendencia más concentrada hacia calificaciones altas (≥4.5).
# 
# Algunos destinos con más de 5000 visitas también tienen calificaciones altas (≥4.5). Sin embargo, a medida que aumenta el número de visitas, parece haber una mayor dispersión en las calificaciones promedio y consideramos que podria ser por mayor diversidad de opiniones en destinos muy visitados o por experiencias más variadas en estos destinos.
# 
# Es importante recalcar que  los destinos con calificaciones promedio bajas (<3.5) tienden a tener un número relativamente bajo de visitas, lo que podría indicar que los destinos menos populares son también aquellos con experiencias menos satisfactorias.
# 
# Esto siempre asumiendo el caso promedio, puesto que siempre existiran outliers con alto flujo de visitas pero bajo de reviews, por ejemplo tiendas de comida rapida, mcdonals, BK, etc.

# #### ¿Algunas categorías tienen consistentemente mejores calificaciones que otras?

# Calificaciones promedio por categoría
calificaciones_por_categoria = df_base.groupby('category')['stars'].mean().reset_index().sort_values(by='stars', ascending=False)

# Visualización
plt.figure(figsize=(10, 6))
plt.bar(calificaciones_por_categoria['category'], calificaciones_por_categoria['stars'], color='skyblue', edgecolor='black')
plt.title("Calificaciones Promedio por Categoría")
plt.xlabel("Categoría")
plt.ylabel("Calificación Promedio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# `Natural_feature"` y `"Tourist_attraction"` tienen calificaciones promedio cercanas a 5.0, por lo que vemos que  los destinos relacionados con naturaleza y turismo son muy valorados. **Podría estar relacionado con experiencias únicas que generan satisfacción.**
# 
# `"Store"`, `"Grocery_or_supermarket"`, y `"School"` tienen calificaciones promedio más bajas, lo que **podría deberse a que estas categorías no generan experiencias emocionales significativas**, ya que están más relacionadas con actividades cotidianas.
# 
# Las categorías relacionadas con ocio, naturaleza y turismo tienden a ser más valoradas que aquellas relacionadas con actividades funcionales o comerciales.

# #### ¿Ciertas ciudades tienen calificaciones consistentemente más altas o bajas?

# Calificaciones promedio por ciudad
calificaciones_por_ciudad = df_base.groupby('city')['stars'].mean().reset_index().sort_values(by='stars', ascending=False)

# Visualización
plt.figure(figsize=(12, 6))
plt.bar(calificaciones_por_ciudad['city'], calificaciones_por_ciudad['stars'], color='orange', edgecolor='black')
plt.title("Calificaciones Promedio por Ciudad")
plt.xlabel("Ciudad")
plt.ylabel("Calificación Promedio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# #### Tendencia en el tiempo.

# Cantidad de visitas por año
visitas_por_anio = df_base.groupby('year')['place_name'].count().reset_index(name='cantidad_visitas')

# Visualización
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(visitas_por_anio['year'], visitas_por_anio['cantidad_visitas'], marker='o', linestyle='-', color='skyblue', label='Cantidad de Visitas')
plt.title("Cantidad de Visitas por Año")
plt.xlabel("Año")
plt.ylabel("Cantidad de Visitas")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# - Antes de 2016, la cantidad de visitas era muy baja, casi insignificante.
# 
# - A partir de 2016, se observa un crecimiento exponencial en la cantidad de visitas, alcanzando un máximo en 2019.
# 
# - En 2019, se registra el mayor número de visitas en todo el periodo analizado, con más de 80,000 visitas.
# 
# - En 2020, la cantidad de visitas cae significativamente. Este descenso se intuye qu eesta relacionado con la pandemia COVID-19.
# 
# - En 2021 y 2022, se observa una recuperación progresiva, aunque no alcanza los niveles de 2019. Esto podría deberse a una lenta reactivación del turismo tras la pandemia.
# 
# - En los años más recientes (2023 y 2024), parece haber una tendencia descendente nuevamente. Esto podría indicar que factores por ejemplos economicos a nivel pais pudieran estar influyendo en el turismo.
# 

# Top 3 categorías por año
top_categorias_por_anio = df_base.groupby(['year', 'category'])['place_name'].count().reset_index(name='cantidad_visitas')
top_categorias_por_anio = top_categorias_por_anio.sort_values(by=['year', 'cantidad_visitas'], ascending=[True, False])

# Filtrar el Top 3 de categorías para cada año
top_3_por_anio = top_categorias_por_anio.groupby('year').head(3)

# Visualización
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(data=top_3_por_anio, x='year', y='cantidad_visitas', hue='category', palette='tab10')
plt.title("Top 3 Categorías por Año")
plt.xlabel("Año")
plt.ylabel("Cantidad de Visitas")
plt.legend(title='Categoría', loc='upper right')
plt.tight_layout()
plt.show()


# - 2018-2019: Durante estos años de mayor actividad, las categorías "restaurant", "tourist_attraction", y "park" dominaron en términos de visitas. Esto sugiere que estas categorías son consistentes en su popularidad.
# 
# - En 2020, como era de esperarse, las visitas caen drásticamente en todas las categorías, debido al impacto de la pandemia de COVID-19. Las categorías más recreativas (como "restaurant" y "tourist_attraction") parecen haber sido las más afectadas.
#   
# - En los años posteriores (2021-2022), las visitas comienzan a recuperarse. "Restaurant" vuelve a liderar, lo que podría deberse a la reactivación del sector gastronómico. "Park" se mantiene como una de las categorías más visitadas, probablemente porque los espacios al aire libre fueron preferidos durante el periodo post-pandemia.
# 
# 
# **Otras observaciones:**
# 
# - Aunque "restaurant" y "tourist_attraction" han sido categorías constantes en el Top 3, otras categorías, como "park" y "point_of_interest", han alternado en popularidad.
# 
# - Es interesante notar la aparición de categorías como "casino" y "museum" en ciertos años, lo que podría reflejar eventos específicos o preferencias locales.
# 
# - En 2023 y 2024, las visitas generales han disminuido en comparación con los años anteriores, pero "restaurant" sigue liderando.
# 
# - La categoría "park" muestra una caída significativa en comparación con su pico en 2018-2019.

# ## 5. Red Bipartita

# Una red bipartita tiene dos tipos de nodos: usuarios y destinos turísticos, para este caso. Los enlaces entre estos nodos representan interacciones (visitas, reviews) y tienen pesos.
# 
# - Nodos del primer conjunto: Usuarios (id_usuario).
# - Nodos del segundo conjunto: Destinos turísticos (place_name)
# 
# **Pesos de los enlaces: Cantidad de visitas/reviews de un usuario a un destino.**
# 
# El objetivo en este análisis es poder modelar las interacciones de la red para poder así generar un sistema de recomendaciones para futuros usuarios.

usuarios = df_base.groupby('id_usuario').agg(count=('place_id','count')).reset_index()
usuarioslista = list(usuarios['id_usuario'][usuarios['count']>1])
red_datos =df_base[df_base['id_usuario'].isin(usuarioslista)]  


# Se considerará pa la información del id del usduario y del lugar
red_datos =red_datos[['id_usuario','place_id']]
#Todas las combinaciones del cliente (primera columna) y las distintas combinaciones de lugares i y j
red_datos = red_datos.merge(red_datos, on='id_usuario', how='outer')
red_datos.columns = [0,'i', 'j'] 
df_agg = red_datos.groupby(['i', 'j']).agg(pairs =('j','count')) # recordar que están duplicados



#Generamos la lista de nodos para diferenciar cada lugar, Con esto obtenemos un ID distintivo para cada lugar
nodes = pd.concat([df_agg.reset_index()['i'], df_agg.reset_index()['j']]).drop_duplicates().sort_values()\
        .to_frame('lugar').reset_index(drop=True).reset_index().rename(columns={'index' : 'id'})


# Lista de Coocurrencia
# Pasamos a trabajar sobre los IDs generados en el punto anterior
list_cooc = (df_agg.reset_index().merge(nodes[['lugar','id']], left_on='i', right_on='lugar')
            .drop(columns=['i', 'lugar']).rename(columns={'id' : 'i'})
            .merge(nodes[['lugar', 'id']], left_on='j', right_on='lugar')
            .drop(columns=['j', 'lugar']).rename(columns={'id' : 'j'})
            [['i', 'j', 'pairs']].sort_values(['i', 'j']))

# Se filtran todos los pesos con valor 0
list_cooc_2 = list_cooc[list_cooc['pairs']>0]


# #### Red base sin Filtrar

# Crear un grafo bipartito
Base = nx.Graph()

# Agregar nodos de usuarios y destinos
usuarios = df_base['id_usuario'].unique()
destinos = df_base['place_name'].unique()

Base.add_nodes_from(usuarios, bipartite=0, tipo='usuario')  # Nodos de tipo 'usuario'
Base.add_nodes_from(destinos, bipartite=1, tipo='destino')  # Nodos de tipo 'destino'

# Agregar enlaces con pesos según el número de visitas
for _, row in df_base.iterrows():
    usuario = row['id_usuario']
    destino = row['place_name']
    if Base.has_edge(usuario, destino):
        Base[usuario][destino]['weight'] += 1
    else:
        Base.add_edge(usuario, destino, weight=1)

# Resumen de la red
print(f"Número total de nodos: {len(Base.nodes)}")
print(f"Número total de enlaces: {len(Base.edges)}")


# La red tiene una gran cantidad de nodos y enlaces que es demasiado grande para representarla gráficamente sin filtrar.
# Por este motivo, es que exploraremos los destinos más visitados y crearemos una subred más manejable para poder entender tendencias y patrones.

# Top destinos más visitados (grado más alto)
destinos_total = sorted(
    [(nodo, grado) for nodo, grado in Base.degree() if Base.nodes[nodo]['tipo'] == 'destino'],
    key=lambda x: x[1],
    reverse=True
)



# Filtrar nodos con grado mayor a un umbral 
umbral_grado = 0
nodos_relevantes = [n for n, grado in Base.degree() if grado > umbral_grado]
subred = Base.subgraph(nodos_relevantes)

# Crear posición específica para nodos bipartitos
usuarios = [n for n, d in subred.nodes(data=True) if d['tipo'] == 'usuario']
destinos = [n for n, d in subred.nodes(data=True) if d['tipo'] == 'destino']
pos = nx.bipartite_layout(subred, usuarios)

# Diferenciar nodos por tipo (colores)
node_colors = ['blue' if n in usuarios else 'orange' for n in subred.nodes()]


# Visualización
plt.figure(figsize=(14, 10))
nx.draw(
    subred,
    pos,
    with_labels=False,
    node_size=20,
    node_color=node_colors,
    edge_color='gray',
    alpha=0.7
)
plt.title("Red Bipartita: Usuarios y Destinos Relevantes")
plt.show()


# #### Proyección Unimodal

W = nx.Graph()
for row in list_cooc_2.itertuples():
    W.add_edge(row.i, row.j, weight=row.pairs)


nx.write_gexf(W, "Red_sin_Filtrar.gexf")


dict10=dict(W.degree())  # node 0 has degree 1
sorted_dict0 = {}
sorted_keys0 = sorted(dict10, key=dict10.get)  # [1, 3, 2]

for w in sorted_keys0:
    sorted_dict0[w] = dict10[w]


fig, ax = plt.subplots(1, 1, figsize=(16, 8))

values = list(sorted_dict0.values())
n, bins, patches = plt.hist(values, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# Make one bin stand out   
patches[47].set_alpha(1) # Set opacity
# Add title and labels with custom font sizes
plt.title('Distribución de grado de la Red', fontsize=12)
plt.xlabel('Grado', fontsize=20)
plt.ylabel('Cantidad de Nodos', fontsize=20)
ax.set_title("Distribución de Grado de la Red",
             pad=24, fontweight=700, fontsize=20)
plt.show()


#F0.edges(data=True)
N10 = len(W)
L10 = W.size()
degrees10 = list(dict(W.degree()).values())
kmin10 = min(degrees10)
kmax10 = max(degrees10)
print("Número de nodos: ", N10)
print("Número de enlaces: ", L10)
print('-------')
print("Grado promedio: ", 2*L10/N10) 
print('-------')
print("Grado mínimo: ", kmin10)
print("Grado máximo: ", kmax10)
print('-------')
print('Densidad: ', nx.density(W))
print('Diametro: ',nx.diameter(W))

nx.draw_networkx(W)


d = {}
for i, j in dict(nx.degree(W)).items():
    if j in d:
        d[j] += 1
    else:
        d[j] = 1
x = np.log10(list((d.keys())))
y = np.log10(list(d.values()))
plt.scatter(x, y, alpha=0.9)
plt.show()


# ### A. Subconjunto: Usuarios más activos y sus destinos favoritos
# Objetivo: Identificar los destinos más frecuentados y bien calificados por los usuarios más activos, ya que estos representan un perfil confiable para futuras recomendaciones.

# Contar la cantidad de destinos únicos visitados por cada usuario
usuarios_activos = df_base.groupby('id_usuario')['place_name'].nunique()

# Ordenar por cantidad de destinos visitados y seleccionar el top 100
top_usuarios_activos = usuarios_activos.sort_values(ascending=False).head(100)

# Mostrar el resultado
print("Top 50 usuarios más activos y destinos únicos visitados:")
print(top_usuarios_activos)

# Visualización del top 100
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
top_usuarios_activos.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 50 Usuarios Más Activos")
plt.xlabel("ID de Usuario")
plt.ylabel("Cantidad de Destinos Únicos Visitados")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()


# Filtrar usuarios que han visitado más de 45 destinos únicos
usuarios_activos = df_base.groupby('id_usuario')['place_name'].nunique()
usuarios_activos = usuarios_activos[usuarios_activos > 45].index

# Crear el subconjunto de datos
subconjunto_usuarios_activos = df_base[df_base['id_usuario'].isin(usuarios_activos)]

print(f"Total de usuarios más activos: {len(usuarios_activos)}")
print(f"Total de filas en el subconjunto: {subconjunto_usuarios_activos.shape[0]}")


# Crear un grafo bipartito
B = nx.Graph()

# Agregar nodos de usuarios y destinos
usuarios = subconjunto_usuarios_activos['id_usuario'].unique()
destinos = subconjunto_usuarios_activos['place_name'].unique()

B.add_nodes_from(usuarios, bipartite=0, tipo='usuario')  # Nodos de tipo 'usuario'
B.add_nodes_from(destinos, bipartite=1, tipo='destino')  # Nodos de tipo 'destino'

# Agregar enlaces con pesos según el número de visitas
for _, row in subconjunto_usuarios_activos.iterrows():
    usuario = row['id_usuario']
    destino = row['place_name']
    if B.has_edge(usuario, destino):
        B[usuario][destino]['weight'] += 1
    else:
        B.add_edge(usuario, destino, weight=1)

# Resumen de la red
print(f"Número total de nodos: {len(B.nodes)}")
print(f"Número total de enlaces: {len(B.edges)}")


# **Datos generales de la red bipartita:**
# 
# - Usuarios más activos considerados: 107 usuarios que han visitado al menos un número significativo de destinos (más de 45).
# - Destinos: 864 destinos únicos en el subconjunto.
# - Total de filas: 6.221 conexiones en el subconjunto.
# - Número de nodos: 971 (usuarios + destinos).
# - Número de enlaces: 6.215 conexiones entre usuarios y destinos.

# La red tiene una gran cantidad de nodos y enlaces que es demasiado grande para representarla gráficamente sin filtrar.
# Por este motivo, es que exploraremos los destinos más visitados y crearemos una subred más manejable para poder entender tendencias y patrones.

# Top destinos más visitados (grado más alto)
top_destinos = sorted(
    [(nodo, grado) for nodo, grado in B.degree() if B.nodes[nodo]['tipo'] == 'destino'],
    key=lambda x: x[1],
    reverse=True
)[:10]

print("Top 10 destinos más visitados en la red bipartita:")
for destino, conexiones in top_destinos:
    print(f"{destino}: {conexiones} conexiones")


# Se filtran los nodos más relevantes para trabajar una subred:

# Filtrar nodos con grado mayor a un umbral 
umbral_grado = 50
nodos_relevantes = [n for n, grado in B.degree() if grado > umbral_grado]
subred = B.subgraph(nodos_relevantes)

# Crear posición específica para nodos bipartitos
usuarios = [n for n, d in subred.nodes(data=True) if d['tipo'] == 'usuario']
destinos = [n for n, d in subred.nodes(data=True) if d['tipo'] == 'destino']
pos = nx.bipartite_layout(subred, usuarios)

# Diferenciar nodos por tipo (colores)
node_colors = ['blue' if n in usuarios else 'orange' for n in subred.nodes()]

# Visualización
plt.figure(figsize=(14, 10))
nx.draw(
    subred,
    pos,
    with_labels=False,
    node_size=20,
    node_color=node_colors,
    edge_color='gray',
    alpha=0.7
)
plt.title("Red Bipartita: Usuarios y Destinos Relevantes")
plt.show()


# El gráfico muestra una red bipartita donde los nodos azules representan a los usuarios más activos (top 50) y los nodos naranjas representan los destinos que han visitado. Cada línea (enlace) indica una relación entre un usuario y un destino, basada en las visitas registradas.

# Filtrar enlaces con peso mayor a 24
proyeccion_usuarios = nx.bipartite.weighted_projected_graph(B, usuarios)
umbral_peso = 24
proyeccion_filtrada = nx.Graph(
    (u, v, d) for u, v, d in proyeccion_usuarios.edges(data=True) if d['weight'] > umbral_peso
)

# Visualizar la red proyectada filtrada
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(proyeccion_filtrada, seed=42)
nx.draw(
    proyeccion_filtrada,
    pos,
    with_labels=False,
    node_size=25,
    node_color='blue',
    edge_color='gray',
    alpha=0.7,
    width=[d['weight'] * 0.1 for (u, v, d) in proyeccion_filtrada.edges(data=True)]
)
plt.title("Proyección Unimodal Filtrada: Relación entre Usuarios")
plt.show()


# **COEFICIENTE DE CLUSTERING**

# Calcular grado de los nodos
grados = dict(proyeccion_filtrada.degree())
# Ordenar por grado descendente
grados_ordenados = sorted(grados.items(), key=lambda x: x[1], reverse=True)


valores = [value[1] for value in grados_ordenados]

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

n, bins, patches = plt.hist(valores, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# Make one bin stand out   
patches[47].set_alpha(1) # Set opacity
# Add title and labels with custom font sizes
plt.title('Distribución de grado de la Red ', fontsize=12)
plt.xlabel('Grado', fontsize=20)
plt.ylabel('Cantidad de Nodos', fontsize=20)
ax.set_title("Distribución de Grado de la Red Filtrada por peso",
             pad=24, fontweight=700, fontsize=20)
plt.show()


#F0.edges(data=True)
N10 = len(proyeccion_filtrada)
L10 = proyeccion_filtrada.size()
degrees10 = list(dict(proyeccion_filtrada.degree()).values())
kmin10 = min(degrees10)
kmax10 = max(degrees10)
print("Número de nodos: ", N10)
print("Número de enlaces: ", L10)
print('-------')
print("Grado promedio: ", 2*L10/N10) 
print('-------')
print("Grado mínimo: ", kmin10)
print("Grado máximo: ", kmax10)
print('-------')
print('Densidad: ', nx.density(proyeccion_filtrada))
print('Diametro: ',nx.diameter(proyeccion_filtrada))

nx.draw_networkx(proyeccion_filtrada)


# **CENTRALIDAD DE INTERMEDIACIÓN**

# Calcular centralidad de intermediación
intermediacion = nx.betweenness_centrality(proyeccion_filtrada)
# Ordenar por centralidad descendente
intermediacion_ordenada = sorted(intermediacion.items(), key=lambda x: x[1], reverse=True)
print("Top 10 nodos por centralidad de intermediación:")
for nodo, valor in intermediacion_ordenada[:10]:
    print(f"Usuario: {nodo}, Centralidad: {valor:.4f}")


# La centralidad de intermediación mide cuántas veces pasa un nodo por los caminos más cortos entre otros nodos. Identifica nodos clave para la estructura de la red. Estos usuarios con alta centralidad de intermediación actúan como puentes entre diferentes grupos de usuarios con destinos comunes.

# **COMUNIDADES**

from networkx.algorithms.community import greedy_modularity_communities

# Detección de comunidades
comunidades = list(greedy_modularity_communities(proyeccion_filtrada))
print(f"Se identificaron {len(comunidades)} comunidades.")
for i, comunidad in enumerate(comunidades[:5]):  # Mostrar las 5 comunidades más grandes
    print(f"Comunidad {i+1}: {len(comunidad)} nodos")


from matplotlib import cm
import numpy as np

colores = cm.rainbow(np.linspace(0, 1, len(comunidades)))
color_map = {}
for i, comunidad in enumerate(comunidades):
    for nodo in comunidad:
        color_map[nodo] = colores[i]

# Dibujar red con colores para cada comunidad
plt.figure(figsize=(12, 8))
nx.draw(
    proyeccion_filtrada,
    pos,
    with_labels=False,
    node_size=50,
    node_color=[color_map[nodo] for nodo in proyeccion_filtrada.nodes],
    edge_color='gray',
    alpha=0.7
)
plt.title("Comunidades en la Red Unimodal de Usuarios")
plt.show()


# Las comunidades reflejan usuarios que tienden a visitar destinos comunes o que comparten intereses específicos.
# 
# Es importante notar que su bien las comunidades estan separadas existen usuarios que funcionan como puente entre las distintas comunides, estos usuarios tienden a realizar reviews de diversos destinos turisticos.

# Obtener enlaces con mayor peso
pesos = nx.get_edge_attributes(proyeccion_filtrada, 'weight')
pesos_ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
print("Top 10 enlaces por peso:")
for (nodo1, nodo2), peso in pesos_ordenados[:10]:
    print(f"Enlace: {nodo1} - {nodo2}, Peso: {peso}")


# Las conexiones más fuertes reflejan pares de usuarios que frecuentan destinos similares, lo que sugiere afinidades en preferencias de viaje. Este dato podría ser utilizado para recomendaciones colaborativas.

from community import community_louvain

# Detectar comunidades en la red unimodal filtrada
particion = community_louvain.best_partition(proyeccion_filtrada)

# Mostrar un ejemplo de cómo luce la partición
print(list(particion.items())[:5])  # Muestra los primeros 5 nodos con sus comunidades


# Asociar comunidades a usuarios
df_comunidades = pd.DataFrame({'id_usuario': list(particion.keys()), 'comunidad': list(particion.values())})

# Unir datos de comunidad con el dataframe original
df_comunidades_merge = pd.merge(df_base, df_comunidades, on='id_usuario')

# Categorías predominantes por comunidad
categorias_por_comunidad = df_comunidades_merge.groupby(['comunidad', 'category'])['place_name'].count().reset_index(name='conteo')

# Calcular el porcentaje de visitas por categoría dentro de cada comunidad
categorias_por_comunidad['porcentaje'] = categorias_por_comunidad.groupby('comunidad')['conteo'].transform(lambda x: (x / x.sum()) * 100)

# Mostrar los resultados para las primeras comunidades
categorias_por_comunidad.sort_values(['comunidad', 'conteo'], ascending=[True, False]).head(10)


# Calificaciones promedio por comunidad
calificaciones_por_comunidad = df_comunidades_merge.groupby('comunidad')['stars'].mean().reset_index()
calificaciones_por_comunidad.rename(columns={'stars': 'calificacion_promedio'}, inplace=True)

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(data=calificaciones_por_comunidad, x='comunidad', y='calificacion_promedio')
plt.title("Calificación Promedio por Comunidad")
plt.xlabel("Comunidad")
plt.ylabel("Calificación Promedio")
plt.xticks(rotation=0)
plt.show()


# En base a la comunidad a la que pertenece cada usuario se crea un sistema de recomendacion de destinos turisticos que no han sido visitados.

# Obtener destinos más visitados por cada comunidad
destinos_por_comunidad = df_comunidades_merge.groupby(['comunidad', 'place_name'])['id_usuario'].count().reset_index(name='conteo')
destinos_por_comunidad = destinos_por_comunidad.sort_values(['comunidad', 'conteo'], ascending=[True, False])

# Crear una lista de recomendaciones para cada usuario
recomendaciones = {}
for usuario, comunidad in df_comunidades.values:
    destinos_visitados = df_base[df_base['id_usuario'] == usuario]['place_name'].unique()
    destinos_comunidad = destinos_por_comunidad[destinos_por_comunidad['comunidad'] == comunidad]['place_name']
    recomendaciones[usuario] = list(set(destinos_comunidad) - set(destinos_visitados))

# Mostrar recomendaciones para los primeros 5 usuarios
for usuario, destinos in list(recomendaciones.items())[:5]:
    print(f"Usuario {usuario}: Recomendaciones: {destinos[:5]}")


# #### B. Subconjunto filtrado por correlación

# Esta sección presenta un análisis detallado de la matriz de co-ocurrencia.
# 
# Primero, se transforma el DataFrame `list_cooc` en un formato de matriz, donde tanto las filas como las columnas representan los distintos elementos. Los valores en las celdas de la matriz indican la frecuencia de co-ocurrencia entre los elementos. Los elementos diagonales, que representan los auto-enlaces, son descartados (ajustados a cero) para excluirlos del análisis.
# 
# En segundo lugar, se crean varias matrices vacías (`phi`, `tval`, `pval`, `phi2` y `exit`) con el propósito de almacenar diferentes medidas estadísticas. La matriz `phi` se utiliza para almacenar los coeficientes de correlación phi, `tval` para los t-values, `pval` para los valores p (que indican la significancia estadística), `phi2` almacena los pesos finales, y `exit` es una matriz binaria que marca si una correlación específica es significativa según los umbrales definidos por `pval_limit` y `cnt_limit`.
# 
# A continuación, se ejecuta un bucle anidado para calcular los coeficientes de correlación phi entre cada par de elementos. Este coeficiente se calcula tomando en cuenta el número total de observaciones y las frecuencias de co-ocurrencia de los elementos. En los casos en que el denominador de la fórmula es cero, el coeficiente phi se asigna a cero para evitar errores de división por cero. Después, se calculan los valores t y p para evaluar la significancia de las correlaciones. Si una correlación cumple con los criterios de significancia establecidos (es decir, si el valor p es menor que `pval_limit` y el número de co-ocurrencias es mayor que `cnt_limit`), la matriz `exit` marca esa correlación como significativa (valor de 1), y su coeficiente phi se almacena en la matriz `phi2`.
# 
# Una vez calculadas las medidas estadísticas, se construyen DataFrames para almacenar los coeficientes phi y los valores p. Estos DataFrames se combinan con los datos originales de co-ocurrencia para formar un DataFrame final, el cual incluye tanto el coeficiente phi como el valor p y el número de co-ocurrencias entre cada par de elementos.
# 
# Finalmente, el análisis visualiza la estructura de la red en función de diferentes umbrales de valores p. Se genera un rango de valores p y se filtra el DataFrame final para incluir únicamente aquellos pares con valores p inferiores a cada umbral. Luego, se calcula y grafica el número de nodos únicos y aristas, mostrando cómo cambia la estructura de la red a medida que se ajustan los niveles de significancia. Los gráficos incluyen líneas verticales que indican umbrales específicos de valores p como referencia para facilitar la interpretación de los resultados.

from tqdm import tqdm
from scipy import stats
# Matriz de Co-Ocurrencia
od_matrix = list_cooc.set_index(['i', 'j']).unstack().fillna(0)
indices = od_matrix.index
columns = od_matrix.columns
od_matrix = od_matrix.values

# eliminamos la diagonal (decisión de análisis: no nos interesan los auto-enlaces)
np.fill_diagonal(od_matrix, 0)


n = len(od_matrix)
phi = np.zeros((n,n)) #matrix to store the phi correlations
tval = np.zeros((n,n))
pval = np.zeros((n,n)) # Significancia
phi2 = np.zeros((n,n)) # Pesos finales
exit = np.zeros((n,n)) # Matriz con unos



M = od_matrix.sum(axis=1) # total number of obervations of each JUEGO (suma sobre las filas)
T = sum(M) # Total number of observations

pval_limit = 1 # significancia limite, 1 significa no filtrar
cnt_limit = 0 # minimo numero de coincidencias a considerar en la od_matrix, 0 significa no filtrar


# El mitico doble for
for i in tqdm(range(n), desc="Calculando Correlaciones"):# https://github.com/tqdm/tqdm  (barra de progreso)
    for j in range(n):
        numerador = T * od_matrix[i, j] - M[i] * M[j]
        denominador = np.sqrt(M[i] * M[j] * (T - M[i]) * (T - M[j]))
        phi[i, j] = numerador / denominador if denominador != 0 else 0
            
        if phi[i, j] != 0:
            sqrt_arg = max(0, max(M[i], M[j]) - 2)
            t_numerador = phi[i, j] * np.sqrt(sqrt_arg)
            t_denominador = np.sqrt(1 - phi[i, j]**2)
            tval[i, j] = t_numerador / t_denominador if t_denominador != 0 else 0
            pval[i, j] = stats.t.sf(np.abs(tval[i, j]), max(M[i], M[j]) - 2) * 2
            # Se evaluan los criterios para ver si la correlacion es significativa
            if pval[i, j] <= pval_limit and od_matrix[i, j] > cnt_limit:
                exit[i, j] = 1
                phi2[i, j] = phi[i, j]
            else:
                exit[i, j] = 0
                phi2[i, j] = 0

# Creamos un dataframe con toda la inforamcion estadistica de la red

df_edgelist_phi  = pd.DataFrame(phi, index=indices, columns=columns).stack(future_stack=True).reset_index()
df_edgelist_phi.columns = ['source', 'target', 'phi']# esto es por convención, recuerden que es este grafo es no-dirigido
df_edgelist_phi = df_edgelist_phi[df_edgelist_phi['source']<=df_edgelist_phi['target']] # eliminamos duplicados

df_edgelist_pval = pd.DataFrame(pval, index=indices, columns=columns).stack(future_stack=True).reset_index()
df_edgelist_pval.columns = ['source', 'target', 'pval']
df_edgelist_pval = df_edgelist_pval[df_edgelist_pval['source']<=df_edgelist_pval['target']] # eliminamos duplicados


# El DataFrame final contiene el valor phi, el pval y el número de veces que ocurre que dos IDs i y j están conectados
df = df_edgelist_phi.merge(df_edgelist_pval).merge(list_cooc.rename(columns={'i': 'source', 'j': 'target'}), how='left')
df['pairs'] = df['pairs'].fillna(0).astype(int) 
df = df[df['pairs']>0]


# Visualización del número de nodos y aristas según un filtro en pval
pvalues = np.arange(0.01,1.01,0.01)#Return evenly spaced values within a given interval.
total_nodes, total_edges = [], []

for p in pvalues:
    __df = df.pipe(lambda x : x[x['pval']<=p])
    n_nodes = pd.concat([__df['source'], __df['target']]).nunique()
    total_nodes.append(n_nodes)
    total_edges.append(len(__df))
pval_df = pd.DataFrame({'pval': pvalues, 'nodes': total_nodes, 'edges': total_edges})

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
pval_df.plot('pval', 'nodes', ax=ax1)
pval_df.plot('pval', 'edges', ax=ax2)
for pval in [0.1, 0.5, 0.8]:
    ax1.axvline(pval, ls='--', color='k')
    ax2.axvline(pval, ls='--', color='k')

ax2.set_yscale('log')
ax1.set_title("Numero de nodos segun corte en pval")
ax2.set_title("Numero de aristas segun corte en pval")
ax1.grid()
ax2.grid()


# Filtros
pval_filt=0.1
df_viz=df[df['pval']<=pval_filt]
df_viz=df_viz[df_viz['source']!=df_viz['target']]
df_viz


# Creación de la red
G=nx.from_pandas_edgelist(df_viz, 'source', 'target')

# The network has disconnected components, let's check the size of each component in descending order
compon=[len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
compon
print('Non connected: '+ str(sum(compon)-compon[0]))
print('Connected nodes en la compoenente gigante: '+ str(compon[0]) +' de '+str(G.number_of_nodes())+' nodos totales.')

print(compon)



# Let's extract and use only the giant component
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]



valores = [value[1] for value in grados_ordenados]

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

n, bins, patches = plt.hist(valores, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# Make one bin stand out   
patches[47].set_alpha(1) # Set opacity
# Add title and labels with custom font sizes
plt.title('Distribución de grado de la Red ', fontsize=12)
plt.xlabel('Grado', fontsize=20)
plt.ylabel('Cantidad de Nodos', fontsize=20)
ax.set_title("Distribución de Grado de la Red Filtrada por peso",
             pad=24, fontweight=700, fontsize=20)


# Filtered
F = S[0]
dict1=dict(F.degree())  # node 0 has degree 1
sorted_dict = {}
sorted_keys = sorted(dict1, key=dict1.get)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = dict1[w]
values = list(sorted_dict.values())

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

n, bins, patches = plt.hist(values, bins=50, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# Make one bin stand out   
# patches[47].set_alpha(1) # Set opacity
# Add title and labels with custom font sizes
plt.title('Distribución de grado de la Red ', fontsize=12)
plt.xlabel('Grado', fontsize=20)
plt.ylabel('Cantidad de Nodos', fontsize=20)
ax.set_title("Distribución de Grado de la Red Filtrada por pvalue",
             pad=24, fontweight=700, fontsize=20)


N10 = len(F)
L10 = F.size()
degrees10 = list(dict(F.degree()).values())
kmin10 = min(degrees10)
kmax10 = max(degrees10)
print("Número de nodos: ", N10)
print("Número de enlaces: ", L10)
print('-------')
print("Grado promedio: ", 2*L10/N10) 
print('-------')
print("Grado mínimo: ", kmin10)
print("Grado máximo: ", kmax10)
print('-------')
print('Densidad: ', nx.density(F))
print('Diametro: ',nx.diameter(F))

nx.draw_networkx(F)


d = {}
for i, j in dict(nx.degree(F)).items():
    if j in d:
        d[j] += 1
    else:
        d[j] = 1
x = np.log10(list((d.keys())))
y = np.log10(list(d.values()))
plt.scatter(x, y, alpha=0.9)
plt.show()


degree_g = dict(nx.degree(F))
clustering_g=dict(nx.clustering(F))

x = degree_g.values()
y = clustering_g.values()


# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, alpha=0.5)
plt.show()



import community.community_louvain as community_louvain

partition = community_louvain.best_partition(F, random_state=5)

# Rest of your code remains the same
partition_df = pd.DataFrame(
    data=partition.items(),
    columns=['id', 'partition']
)

community_sizes = partition_df.groupby('partition')['id'].count().sort_values(ascending=False)

print(f"Number of communities: {len(community_sizes)}")
print("\nCommunity sizes:")
print(community_sizes)


plt.figure(figsize=(20, 10))

partition = community_louvain.best_partition(F)

size = (len(set(partition.values())))#Numero de comunidades
print('Se detectan %d comunidades' % (size))

pos = nx.spring_layout(F) # Layout para la red (coordenadas de los nodos y enlaces)

count = 0
colors = [np.array(cm.jet(x)).reshape(1,-1) for x in np.linspace(0, 1, size)]#cm.jet es el mapa de colores https://www.programcreek.com/python/example/56498/matplotlib.cm.jet
for com in set(partition.values()): #para cada comunidad
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]#guarda los personajes que pertenecen a la comunidad `com`
    nx.draw_networkx_nodes(F, pos, list_nodes, node_size = 100, node_color=colors[count])#plotea nodos con colors por comunidad
    count = count + 1# para iterar sobre los colores
nx.draw_networkx_edges(F, pos, alpha=0.2)#plotea enlaces
plt.show()


# Dentro de las comunidades se obsevan ciertos nodos que funcionan como purta central entre las comunidades. Se podria generar recomendaciones a usuarios de otras comunidades a estos nodos en particular para expandir el mercado




# Asociar comunidades a usuarios
df_comunidades_2 = pd.DataFrame({'N': list(partition.keys()), 'comunidad': list(partition.values())})

# Unir datos de comunidad con el dataframe original
df_comunidades_merge_2 = pd.merge(df_base, df_comunidades_2, on='N')

# Categorías predominantes por comunidad
categorias_por_comunidad_2 = df_comunidades_merge_2.groupby(['comunidad', 'category'])['place_name'].count().reset_index(name='conteo')

# Calcular el porcentaje de visitas por categoría dentro de cada comunidad
categorias_por_comunidad_2['porcentaje'] = categorias_por_comunidad_2.groupby('comunidad')['conteo'].transform(lambda x: (x / x.sum()) * 100)

# Mostrar los resultados para las primeras comunidades
categorias_por_comunidad_2.sort_values(['comunidad', 'conteo'], ascending=[True, False]).head(10)


def agrupar_categorias_principales(df, comunidad, top_n=3):

    # Filtrar el DataFrame para la comunidad especificada
    grupo = df[df['comunidad'] == comunidad]
    
    # Obtener las N categorías principales por porcentaje
    categorias_principales = grupo.nlargest(top_n, 'porcentaje')
    
    # Calcular los valores agregados para 'otros'
    porcentaje_otros = grupo[~grupo.index.isin(categorias_principales.index)]['porcentaje'].sum()
    conteo_otros = grupo[~grupo.index.isin(categorias_principales.index)]['conteo'].sum()
    
    # Convertir las categorías principales a registros
    resultado = categorias_principales.to_dict('records')
    
    # Agregar la categoría 'otros' si es aplicable
    if porcentaje_otros > 0:
        resultado.append({
            'comunidad': comunidad,
            'category': 'otros',
            'conteo': conteo_otros,
            'porcentaje': porcentaje_otros
        })
    
    # Convertir el resultado nuevamente a un DataFrame
    return pd.DataFrame(resultado)


df_aux = categorias_por_comunidad_2.sort_values(['comunidad', 'conteo'], ascending=[True, False])
resultados = []
    
for i in range(5):
    df_procesado = agrupar_categorias_principales(df_aux, i, 3)
    resultados.append(df_procesado)
    
df_aux = pd.concat(resultados, ignore_index=True)


# Pivot the data for stacked bar chart
pivot_df = df_aux.pivot_table(index='comunidad', columns='category', values='porcentaje', fill_value=0)

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(16, 8))

pivot_df.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    colormap='Spectral',  # Choose a colormap for better differentiation
    alpha=1
)

# Add labels, title, and legend
ax.set_title("Distribution of Categories by Community", fontsize=20, pad=20, fontweight='bold')
ax.set_xlabel("Community", fontsize=16)
ax.set_ylabel("Percentage", fontsize=16)
ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()

