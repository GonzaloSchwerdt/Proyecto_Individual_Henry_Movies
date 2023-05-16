#!/usr/bin/env python
# coding: utf-8

'''
##############################################################################################

########################## Proyecto Individual - Gonzalo Schwerdt ############################

##############################################################################################
'''

#Importamos todas las librerias necesarias

import pandas as pd
import numpy as np
import json
import re

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from fastapi import FastAPI
import uvicorn
app = FastAPI()

# importamos el dataset

url = 'https://drive.google.com/uc?id=1Ej4or9KVgP7hVTQghdqTDiIUCnHABJOs&export=download'
df = pd.read_csv(url,index_col=False, names=["adult","belongs_to_collection","budget","genres","homepage","id","imdb_id","original_language","original_title","overview","popularity","poster_path","production_companies","production_countries","release_date","revenue","runtime","spoken_languages","status","tagline","title","video","vote_average","vote_count"],header=0)
df = df.drop(columns=["video","imdb_id","adult","original_title","vote_count","poster_path", "homepage"])

'''
####################################################################

########################## NORMALIZACION ###########################

####################################################################
'''
#Funciones para la normalizacion

# Objetivo: df["genres"] # Traduce el str input entrante a la funcion un diccionario. 
# Por último extrae sus valores internos retornando un string de los mismos separados por ", ".
# Ademas traduce los valores float en vacios para mas tarde eliminarlos.

def get_values(lista): 
    if type(lista) == float:
        return ""
    res=""
    import json
    lista_diccionarios = json.loads(lista.replace("'", "\""))
    for dic in lista_diccionarios:
        res+=str((list(dic.values())[1]))+", "
    return res[:(len(res))-2]

def get_values_countries(lista): # Objetivo: df["production_countries"]. Realiza lo mismo que la anterior funcion
    if type(lista) == float:
        return ""
    if lista == "":
        return ""
    res=""
    import json
    lista_diccionarios = json.loads(lista.replace("'", "\""))
    if type(lista_diccionarios)== float:
        return ""
    for dic in lista_diccionarios:
        res+=str((list(dic.values())[0]))+", "
    return res[:(len(res))-2]


# El objetivo del siguiente bloque es realizar lo mismo que las dos funciones anteriores pero con varias modificaciones por varios motivos.
# Dicho proceso filtra varias filas que no se podian convertir a diccionarios ya que contenian caracteres que generaban errores.
# De todos modos algunos pudieron ser recuperados mediante varias normalizaciones

count = 0
unalista = []
for i, val in enumerate(df["production_companies"]):
    if isinstance(val, str): # Si es una cadena de texto, intentamos convertirla en un objeto JSON
        try:
            json.loads(val.replace("\"","").replace("'", "\""))
        except json.JSONDecodeError:
            unalista.append(i)

unalista = set(unalista)

# Esta funcion normaliza algunas filas especiales que no podrán ser convertidas con al función principal.

def get_values_production_companies_alter(lista): 
    regex = r"'name': '(?:(?<!\\)'|\\')*([^']+)'"
    matches = re.findall(regex, lista)
    if not matches:
        return ""
    return ", ".join(matches)

# Funcion principal que busca extraer y normalizar los elementos de df["production_companies"] para traducirlos en str separados por ", "
# En base a un tratamiento muy especifico de dichos datos. Ejemplo de una fila que da error: # '[{\'name\': \'Castle Rock Entertainment\', \'id\': 97}, {\'name\': "Hell\'s Kitchen Films", \'id\': 2307}]'

def get_value_production_companies(x):
    if x.name in unalista:
        return x["production_companies"]
    if type(x["production_companies"]) != str:
        return x["production_companies"]
    if "\'" in x["production_companies"]:
        return get_values_production_companies_alter(x["production_companies"])
    
    # Si los datos no son especiales ni generadores de errores se aplica una normalizacion/traduccion común de json a dicc a str separados por ", "
    res=""
    lista_diccionarios = json.loads(x["production_companies"].replace("'", "\""))
    for dic in lista_diccionarios:
        res+=str((list(dic.values())[1]))+", "
    return res[:(len(res))-2]

#Retorna el nombre de la collection
def get_collection_name(string):
    if type(string) != str:
        return string
    v1=3
    res=""
    for let in string:
        if let == " ":
            v1-=1
        if v1 <= 0 and let == ",":
            break
        if v1 <= 0:
            res+=let
    return res

# Retorna el año de df["release_year"]
def get_year(lista):
    return lista.year

# Retorna df["return"]
def return_fun(rev,bud):
    if bud==0:
        return 0
    return rev/bud

#Pocos paises no se podian convertir en str debido a errores, por eso se guardan en una lista para luego eliminarlos.
listaCountries=[]
for i, val in enumerate(df["production_countries"]):
    if isinstance(val, str):
        try:
            json.loads(val.replace("\"","").replace("'", "\""))
        except json.JSONDecodeError as e:
            listaCountries.append(i)

# Aplicamos funciones anteriores
df["production_countries"] = df["production_countries"].drop(index=listaCountries)
df["production_countries"]=df["production_countries"].apply(get_values_countries)

# Aplicamos las funciones anteriormente descriptas
df["genres"]=df["genres"].apply(get_values)
df["belongs_to_collection"]=df["belongs_to_collection"].apply(get_collection_name)
df["production_companies"] = df.apply(get_value_production_companies, axis=1)


# Cambiamos variables a integer y rellenamos nulos
df["revenue"] = df["revenue"].fillna(0)
df["revenue"] = df["revenue"].apply(int)
df["budget"] = df["budget"].apply(int)


df = df.dropna(subset=['release_date']) # Eliminamos valores nulos en "release_date"
df['release_date'] = pd.to_datetime(df['release_date']) # Cambiamos la variable de "release_date" a datetime 
df["release_year"]= df['release_date'].apply(get_year) # Aplicamos normalizacion por funcion

# Para no perder datos en base a los nulos de df['production_countries'] decido por crear la clasificacion "UnknownCountry", lo mismo en df['genres'] con "UnknownGenre"
df['belongs_to_collection'] = df['belongs_to_collection'].fillna('')
df['production_countries']=df['production_countries'].replace(" ","UnknownCountry")
df['production_countries']=df['production_countries'].replace("","UnknownCountry")
df['genres']=df['genres'].replace(" ","UnknownGenre")
df['genres']=df['genres'].replace("","UnknownGenre")

# Se crea la columna de retorno y se la normaliza
df["return"]=df['revenue']/df['budget']
df["return"] = df["return"].replace(float("inf"),0)
df["return"] = df["return"].fillna(0)

'''
####################################################################

###################### FUNCIONES PARA LA API #######################

####################################################################
'''

# Peliculas_mes

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes): #'''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes (nombre del mes, en str, ejemplo 'enero') historicamente''' 
    if mes=="":
       return "Ingresa un mes del año"
    cant=0
    if mes == "enero":
        mesNum=1
    elif mes == "febrero":
        mesNum = 2
    elif mes == "marzo":
        mesNum = 3 
    elif mes == "abril":
        mesNum = 4 
    elif mes == "mayo":
        mesNum = 5 
    elif mes == "junio":
        mesNum = 6
    elif mes == "julio":
        mesNum = 7 
    elif mes == "agosto":
        mesNum = 8 
    elif mes == "septiembre":
        mesNum = 9 
    elif mes == "octubre":
        mesNum = 10 
    elif mes == "noviembre":
        mesNum = 11 
    elif mes == "diciembre":
        mesNum = 12 
    else:
        return "Ingresa un mes correcto"
    
    for row, mon in enumerate(df['release_date']):
        try:
            df['release_date'][row]
        except KeyError:
            continue

        if df['release_date'][row].month == mesNum:
            cant = cant + 1

    return 'Mes:',mes, '- Cantidad:',cant


# Peliculas_dia

@app.get('/peliculas_dia/{dia}')
def peliculas_dia(dia): #'''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia (de la semana, en str, ejemplo 'lunes') historicamente''' 
    if dia=="":
       return "Ingresa un dia de la semana"
    cant=0
    if dia == "martes":
        diaNum=1
    elif dia == "miercoles":
        diaNum = 2
    elif dia == "jueves":
        diaNum = 3 
    elif dia == "viernes":
        diaNum = 4 
    elif dia == "sabado":
        diaNum = 5 
    elif dia == "domingo":
        diaNum = 6
    elif dia == "lunes":
        diaNum = 0 
    else:
        return "Ingresa un dia de la semana correcto"

    for row, d in enumerate(df['release_date']):
        try:
            df['release_date'][row]
        except KeyError:
            continue

        if df['release_date'][row].weekday() == diaNum:
            cant = cant + 1
    
    return 'Dia:', dia, '- Cantidad:',cant


# Franquicia

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia): #'''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio''' 
    
    cant,total,ctrl=0,0,False # Ctrl se utiliza en el caso de que no se haya encontrado el input en el dataset
    for row, d in enumerate(df['belongs_to_collection']):
        try:
            df['belongs_to_collection'][row]
        except KeyError:
            continue

        else:
            if franquicia in df['belongs_to_collection'][row]:
                cant = cant + 1
                total+=df['revenue'][row]
                ctrl=True
    if total == 0: # Si el total es 0 entonces la cant es 0 entonces cant lo convertimos en 1 para que no divida por 0.
        cant = 1

    if not ctrl:
        return 'La franquicia no se encuentra en la lista'
    return 'Franquicia:',franquicia, '- Cantidad:',cant, '- Ganancia_total:',int(total), '- Ganancia_promedio:',int(total/cant)


# Peliculas_pais

@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais): #'''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo''' 
    if pais=="":
       return "Ingresa un pais"
    
    cant,ctrl=0,False # Ctrl se utiliza en el caso de que no se haya encontrado el input en el dataset
    for row, p in enumerate(df['production_countries']):
        try:
            df['production_countries'][row]
        except KeyError:
            continue
        else:
            if pais in df['production_countries'][row]:
                cant = cant + 1
                ctrl=True
    if not ctrl:
        return 'El pais no se encuentra en la lista'
    return 'Pais:',pais, '- Cantidad:',cant


# Productoras

@app.get('/productoras/{productora}')
def productoras(productora): #'''Ingresas la productora, retornando la ganancia total y la cantidad de peliculas que produjeron'''
    if productora=="":
       return "Ingresa una productora" 
    cant,total,ctrl=0,0,False # Ctrl se utiliza en el caso de que no se haya encontrado el input en el dataset
    for row, p in enumerate(df['production_companies']):
        try:
            df['production_companies'][row]
        except KeyError:
            continue
        else:
            if productora in df['production_companies'][row]:
                cant = cant + 1
                total+=df['revenue'][row]
                ctrl=True
    if not ctrl:
        return print('La productora no se encuentra en la lista')
    
    return 'Productora:', productora,' - Ganancia_total: ', int(total), '- Cantidad:' , int(cant)


# Retorno

@app.get('/retorno/{pelicula}')
def retorno(pelicula): #'''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo''' 
    if pelicula=="":
       return "Ingresa una pelicula"
    
    cant,total,ctrl=0,0,False # Ctrl se utiliza en el caso de que no se haya encontrado el input en el dataset
    for row, p in enumerate(df['title']):
        
        try:
            df['title'][row]
        except KeyError:
            continue
        else:
            if pelicula in df['title'][row]:
                ctrl = True
                break
    
    # Por la falta de datos en df['revenue'] y df['budget'], muchos de los resultados en df["return"] son nulos o ininitos al divir por 0. Entonces devuelve "Desconocido"
    if int(df['return'][row]) ==0 :
        return 'Pelicula:',pelicula, '- Inversion:',df['budget'][row], '- Ganacia:',df['revenue'][row],'- Retorno:',"Desconocido", '- Anio:',df['release_year'][row]
    
    if not ctrl:
        return 'La pelicula no se encuentra en la lista'
    
    return 'Pelicula:',pelicula, '- Inversion:',int(df['budget'][row]), '- Ganacia:',int(df['revenue'][row]),'- Retorno:',int(round(df['return'][row],2)), '- Anio:',int(df['release_year'][row])

'''
####################################################################

#################### SISTEMA DE RECOMENDACIÓN ######################

####################################################################


# DESCRIPCIÓN: 


El sistema de recomendación tomará de referencia las características del titulo ingresado en la funcion get_similar_movies(title).
Es así como a través de un procesamiento de clustering se encontrarán los 5 Titulos de películas que se asimilen mas 
en torno a dischas características.

He decidido tomar como características mas importantes:

- "runtime": diferenciandose peliculas cortas de largas. 
- "release_year" diferenciándose por sus respectivas epocas variando mucho dichas peliculas.
- "genres" una de las caracteristicas mas importantes a tomar ya que determina en gran medidad el film.
- "production_countries" no solo representa el lugar de origen, sino tambien el idioma original de la misma. (por eso descartada la columna 
    de "original_language")ya que seria redundante

Las demas columnas representan valores de ganancias/perdidas en dinero y valores de popularidad que no deberian ser tenidas en cuenta para caracterizar a las peliculas.

'''

#Importamos el dataframe que estabamos utiliando en un nuevo dataframe llamado "dataMachine" (por Machine Learning)
dataMachine=pd.DataFrame()
dataMachine["title"] = df["title"]
dataMachine["runtime"] = df["runtime"]
dataMachine["release_year"] = df["release_year"]
dataMachine["genres"] = df["genres"]
dataMachine["production_countries"] = df["production_countries"]

# Tokenizamos los distintos paises y generos que existen en el dataset
countries = dataMachine['production_countries'].str.split(', ', expand=True)
genres = dataMachine['genres'].str.split(', ', expand=True)

# Al ser variables cualitativas es necesario generar dummies de las mismas creando una columna para cada una con 
# variables de 0/1 que indican presencia o ausencia de la característica.
country_dummies = pd.get_dummies(countries, prefix='', prefix_sep='')
genres_dummies = pd.get_dummies(genres, prefix='', prefix_sep='')

#Algunas columnas se repiten asi que se las agrupa para no perder datos
country_dummies = country_dummies.groupby(level=0, axis=1).sum().clip(upper=1)
genres_dummies = genres_dummies.groupby(level=0, axis=1).sum().clip(upper=1)

# Se agreaga las columnas de dummies al dataframe
dataMachine = pd.concat([dataMachine, genres_dummies], axis=1)
dataMachine = pd.concat([dataMachine, country_dummies], axis=1)

# Se eliminan las columnas originales de 'production_countries' y 'genres'
dataMachine.drop('production_countries', axis=1, inplace=True)
dataMachine.drop('genres', axis=1, inplace=True)

# Para un postprocesado de datos se necesita rellenar los valores nulos de dichas columnas.
dataMachine["runtime"] = dataMachine["runtime"].fillna(0)
dataMachine['runtime'] = dataMachine['runtime'].apply(int) # Se convierte en integer ya que son "minutos"
dataMachine['runtime'] = dataMachine['runtime'].replace(0,int(dataMachine['runtime'].mean()))
dataMachine["release_year"] = dataMachine["release_year"].fillna(0)
dataMachine['release_year'] = dataMachine['release_year'].apply(int)

# Gracias a un grafico de cajas se puede concluir que los outliers en dataMachine['runtime'] son {min: 60, max:200}
# Asi que son eliminados. Cabe recalcar que un largometraje se considera a partir de los 60 minutos. Además la mayor parte de las peliculas que
# superaban los 200 minutos eran series y no peliculas.
dataMachine = dataMachine.drop(dataMachine[dataMachine['runtime'] > 200].index)
dataMachine = dataMachine.drop(dataMachine[dataMachine['runtime'] < 60].index)


# Al explorar los datos inferí que los titulos duplicados eran en realidad otras versiones de la misma pelicula gracias a que las demas caracteristicas
# eran distintas, como "release" o los "genres". Por eso para diferenciarlas solo a dichas duplciadas les agregue en su nombre su año de "release".

lista_duplicados = []
for n in dataMachine[dataMachine.duplicated(subset=["title"], keep=False)]["title"]:
    if n not in lista_duplicados:
        lista_duplicados.append(n)

def add_release_year(title, release_year):
    if title in lista_duplicados:
        return f"{title} ({release_year})"
    else:
        return title

dataMachine["title"] = dataMachine.apply(lambda x: add_release_year(x["title"], x["release_year"]), axis=1)

# Eliminamos las que si quedaron duplicadas por mas que se les haya agregado el año "release"
dataMachine = dataMachine.drop_duplicates(subset=["title"])

###################

# MODELO DE MACHINE LEARING - CLUSTERING

###################

# Se seleccionan características a estandarizar
X = dataMachine[["runtime","release_year"]]
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Se reempalzan los valores estandarizados en el DataFrame
dataMachine[["runtime","release_year"]] = X_norm

'''
#ENTRENAMIENTO DEL MODELO

DESCRIPCIÓN:

Lo que sucede en el entrenamiento es lo siguiente. Se busca encontrar las 5 peliculas mas similares a la ingresada, en el input de la funcion, desde la
distancia euclidiana entre dichas 5 con la de referencia; a partir del conjunto de caracteristicas que cada una representa. Al utilizar un procesamiento de clustering 
podemos reducir la cantidad de datos procesados a la hora de comparar distancias ya que solo se compararán datos dentro del cluster mismo. 

Resumiendo : Clustering => Distancias Euclidianas entre los valores

'''

# Se define el rango de la X de entrenamiento
X = dataMachine.columns[1:]

# Se determina el numero de clusters. Cabe aclarar que desde el gráfico "Elbow method" el número de cluster óptimo es entre 5 o 6 para
# calcular las distancias euclidianas
n_clusters = 6

# Se crea el modelo KMeans
kmeans = KMeans(n_clusters=n_clusters,n_init=10, random_state=42)

# Se entrena el modelo con los datos normalizados
kmeans.fit(X_norm)

# Se predice el cluster para cada observación
cluster_labels = kmeans.predict(X_norm)

# Se agregan etiquetas de cluster a los datos originales
dataMachine["cluster"] = cluster_labels


# FUNCION DE RECOMENDACION

@app.get('/recomendacion/{title}')
def recomendacion(title):
    
    # Encontrar cluster de la película buscada
    try:
        movie_cluster = dataMachine[dataMachine["title"] == title]["cluster"].iloc[0]
    except IndexError:
        return "La pelicula que ingresaste no esta en la lista o tiene un error ortografico"
    
    movie_cluster = dataMachine[dataMachine["title"] == title]["cluster"].iloc[0]
    
    # Se seleccionan películas del mismo cluster, excepto la buscada
    similar_movies = dataMachine[(dataMachine["cluster"] == movie_cluster) & (dataMachine["title"] != title)].copy()
    
    # Se calcula la distancia euclidiana entre cada película y la de referencia
    reference_movie = dataMachine[dataMachine["title"] == title][X].values
    distances = np.linalg.norm(similar_movies[X].values - reference_movie, axis=1)
    
    # Se agrega la distancia como columna en similar_movies y se ordenan desde la mas cercana hasta la mas lejana.
    similar_movies["distance"] = distances
    similar_movies = similar_movies.sort_values(by=["distance"], ascending=[True])
    
    # Se devuelven las 5 películas más similares
    return 'Lista Recomendada:', similar_movies.head(5)["title"].tolist()

'''
####################################################################

######################### SE INICIA LA API #########################

####################################################################
''' 

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=10000)

#############################################################################################################################################
