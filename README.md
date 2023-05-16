# Proyecto_Individual_Henry

Data-Science - Cohorte 10 - Full Time

Gonzalo Schwerdt

#Video: https://drive.google.com/file/d/1ur7HGb3yg2sz4Fy7Wy8GqGKuLGnhKlym/view?usp=sharing

Se realizaron multiples normalizaciones al data set:
- Primero se extrajo todos los valores de los diccionarios/listas presentes en el dataset.
- En el caso de columnas con valores str, cualquier otro valor inválido se lo reemplaza con un vacío.
- En el caso de columnas numéricas se completó los vacíos con el promedio del conjunto de dichos datos.
- En el caso de países y géneros, si habia un valor vacío, se lo reemplazó por "UnknownCountrie" y "UnknownGenre", para no perder datos.
- Luego para el modelo de machine learning se extrajo solo las columnas necesarias del dataFrame original en uno nuevo para su procesamiento.
- En la columna runtime, a través de un diagrama de cajas se notó como habian cierto valores outliers por debajo de 60 minutos (corto-metrajes) y otros por arriba de 200 minutos (series de television). Los mismos fueron eliminados.
- En la columna de title habian ciertos nombres repetidos, pero luego de un análisis concluí que eran distintas películas con el mismo nombre. Asi que para diferenciarlas les agregué a su nombre su año de estreno.
- En el caso de los países y los géneros se los transformó a columnas dummies por el hecho de ser datos cualitativos. La columna runtime y release_year se estandarizaron para coincidir con todos los datos en general.
- A través del metodo de elbow se pudo ver que el numero óptimo de clusters estaba entre 5 y 6.
