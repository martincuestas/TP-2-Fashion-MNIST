# -*- coding: utf-8 -*-
"""
Trabajo Práctico 2: Fashion-MNIST
Laboratorio de Datos (DC) - Comisión Laje

Nombre/Número del grupo: 9

Estudiantes:
    Cuestas Martín
    Nakasone Julián
    Poli Dante

Código asociado a los distintos procesos relatados en el informe.
Todo es reproducible.

Contenidos:
    1. Importación de librerías (BLOQUE)
    2. Carga de datos (BLOQUE)
    3. Funciones definidas (BLOQUE)
        3.1. Análisis exploratorio
        3.2. Clasificación binaria
        3.3. Clasificación multiclase
    4. Código suelto
        4.1. Análisis exploratorio (BLOQUE)
        4.2. Clasificación binaria (BLOQUE)
        4.3. Clasificación multiclase (BLOQUE)
    
"""

#%% IMPORTACIÓN DE LIBRERIAS

import pandas as pd # Para manejo de datos
from sklearn.tree import DecisionTreeClassifier # Para arbol de decisión
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.metrics import accuracy_score, confusion_matrix # Para exactitud y matriz de confusión
from sklearn.model_selection import StratifiedKFold, cross_val_score # Para kfold cross validation
import matplotlib.pyplot as plt # Para graficar
from sklearn.neighbors import KNeighborsClassifier # Para Knn
import duckdb # Para algebra relacional
from sklearn.preprocessing import LabelEncoder # Para pasar atributos str a int
import seaborn as sns # Para graficar
import numpy as np # Para manipular datos
import itertools

#%% CARGA DE DATOS

datos_fashion = pd.read_csv("Fashion-MNIST.csv")


#%% FUNCIONES DEFINIDAS

# ---- ANALISIS EXPLORATORIO ---- #

def graficar_cantidad_prendas_por_clase():
    # Se manipulan los datos para obtener información de su naturaleza
    print("Muestra:", datos_fashion.head())
    print("Tamaño de la base de datos (Instancias, Atributos):", datos_fashion.shape)
    print("Tipos de atributos:", datos_fashion.dtypes)
    
    # La variable de interes, lo que se va a querer clasificar, es el ultimo atributo: label
    print("Cantidad de prendas diferentes", datos_fashion.label.value_counts())
    
    
    prendas_distintas = duckdb.sql("""
                          SELECT label, COUNT(*) AS Cantidad
                          FROM datos_fashion
                          GROUP BY Label
                          ORDER BY Cantidad DESC
                         """).df()
    
    prendas_distintas['label'] = prendas_distintas['label'].replace(label_map)
    
    sns.barplot(data = prendas_distintas, x ='label', y ='Cantidad') # Quiero ordenar de mayor a menor
    plt.title("Cantidad de prendas")
    plt.xlabel("Prenda")
    plt.ylabel("Cantidad")
    plt.xticks(rotation = 45)
    plt.show()


def graficar_cantidad_instancias_con_info_por_atributo():
    atributos = datos_fashion.drop(columns=['label'], errors='ignore')
    
    conteo_info = (atributos != 0).sum()
    
    conteo_info_df = conteo_info.reset_index()
    conteo_info_df.columns = ['Atributo', 'Instancias_con_info']
    
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(conteo_info)), conteo_info.values, linewidth=0.8)
    plt.title("Cantidad de instancias con información (≠ 0) por atributo")
    plt.xlabel("Índice del atributo")
    plt.ylabel("Cantidad de instancias")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def proyectar_valor_promedio_de_atributos(lista_clases):
    promedios = []

    plt.figure(figsize=(12, 7))

    n_clases = len(lista_clases)

    # Crear un colormap continuo y muestrear 'n_clases' colores
    cmap = plt.cm.get_cmap('viridis', n_clases)
    colors = [cmap(i) for i in range(n_clases)]

    for idx, i in enumerate(lista_clases):
        prenda_instancias = datos_fashion[datos_fashion['label'] == i]
        prenda_instancias = prenda_instancias.drop('Unnamed: 0', axis = 1)
        prenda_prom = prenda_instancias.mean(axis=0).apply(np.floor).astype(int)
        promedios.append(prenda_prom)

        plt.plot(
            range(len(prenda_prom.drop('label'))),
            prenda_prom.drop('label').values,
            label=label_map[i],
            color=colors[idx],
            linewidth=1.5
        )

    plt.xlabel("Atributo")
    plt.ylabel("Valor promedio de atributo (entero)")
    plt.title("Valor promedio de atributos por clase")
    plt.ylim(0, 256)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def proyectar_imagen_promedio_de_clase(clase):
    # Se calculan los promedios de atributo de una clase
    prenda_instancias = datos_fashion[datos_fashion['label'] == clase]
    prenda_prom = prenda_instancias.mean(axis=0).apply(np.floor).astype(int)
    img_nbr = prenda_prom
    img = np.array(img_nbr.iloc[1:-1]).reshape(28,28)
    # Proyección de la imagen promedio
    plt.imshow(img, cmap="gray")
    plt.title(f"Imagen promedio - Clase {clase}")
    plt.axis("on")
    plt.show()


def proyectar_imagen_promedio_total():
    # Se calculan los promedios de atributo de todo el dataset original
    prenda_prom = datos_fashion.mean(axis=0).apply(np.floor).astype(int)
    img_nbr = prenda_prom
    img = np.array(img_nbr.iloc[1:-1]).reshape(28,28)
    # Proyección de la imagen promedio
    plt.imshow(img, cmap="gray")
    plt.title("Imagen promedio - Todas las clases")
    plt.axis("on")
    plt.show()


def proyectar_varianza_clase(clase, vmax_global=8000, cmap='hot'):
    # Cargamos el dataset
    #datos_fashion = pd.read_csv("Fashion-MNIST.csv")

    # Filtramos solo la clase deseada
    datos_clase = duckdb.sql(f"""
        SELECT *
        FROM datos_fashion
        WHERE label = {clase}
    """).df()

    # Obtenemos las columnas de píxeles
    columnas_pixeles = [col for col in datos_fashion.columns if col.startswith('pixel')]
    pixeles = datos_clase[columnas_pixeles]

    # Calculamos la varianza por píxel
    varianza_pixeles = pixeles.var().values.reshape(28, 28)

    # Graficamos el mapa de varianza con escala fija
    plt.figure(figsize=(6, 6))
    plt.title(f"Varianza de píxeles dentro de clase {clase}")
    img = plt.imshow(varianza_pixeles, cmap=cmap, vmin=0, vmax=vmax_global)
    plt.colorbar(img, label='Varianza (escala fija)')
    plt.axis('off')
    plt.show()


# ---- CLASIFICACIÓN BINARIA ---- #

def n_pixeles_importantes(data, n):

  datos_fashion_246 = duckdb.sql(""" SELECT *
                                    FROM datos_fashion
                                    WHERE  label = 8 OR label = 0 """).df()

  pixeles = datos_fashion_246.loc[:, [col for col in datos_fashion.columns if col.startswith('pixel')]]


  labels = datos_fashion_246['label']

  # Promedio por clase para cada pixel
  promedios_por_clase = pixeles.groupby(labels).mean()

  # Varianza entre clases para cada pixel
  varianza_pixeles = promedios_por_clase.var()
  print(f"Top {n} píxeles con mayor variación entre clases:")
  pixeles_mayor_varianza = varianza_pixeles.sort_values(ascending=False).head(n)
  print(pixeles_mayor_varianza)

  # Lista de columnas más importantes 
  pixeles_top = list(pixeles_mayor_varianza.index)  # Pixeles más importantes
  print(f"Lista de columnas top {n}:", pixeles_top)

  # Matriz 28x28
  varianza_matriz = varianza_pixeles.values.reshape(28, 28)

  return varianza_matriz, pixeles_top


def graficar_matriz_varianza(varianza_matriz):
    # Heatmap
    plt.figure(figsize=(8, 8))
    plt.title("Mapa de varianza de píxeles entre clases 0 y 8")
    plt.imshow(varianza_matriz, cmap='hot')
    plt.colorbar()
    plt.show()
  

def performance_KNN(k, distancia, subconjunto):
     x_funcion_train = x_train[subconjunto]
     x_funcion_test = x_test[subconjunto]
     modelo_Knn = KNeighborsClassifier(n_neighbors = k, metric = distancia)
     modelo_Knn.fit(x_funcion_train, y_train)
     y_pred = modelo_Knn.predict(x_funcion_test)
     exactitud = accuracy_score(y_test, y_pred)
     return (f"k = {k}, distancia = {distancia}, exactitud = {exactitud}")


# Para probar con diferentes k vecinos cercanos, se utiliza la siguiente función

def performance_kNN_kVariable(k_max, distancia, subconjunto):
    K_list = [] # Guardo los distintos valores de K nearest neighbours
    exactitudes = [] # Guardo las distintas exactitudes para cada k, que serán el promedio kfolder
    x_funcion_train = x_train[subconjunto]
    x_funcion_test = x_test[subconjunto]
    for k_vecinos in range (1, k_max + 1): # Itero por numero de k vecinos más cercanos
        modelo_Knn = KNeighborsClassifier(n_neighbors = k_vecinos, metric = distancia) # Tomo el número de k más cercanos de la iteración
        modelo_Knn.fit(x_funcion_train, y_train)
        y_pred = modelo_Knn.predict(x_funcion_test)
        exactitud = accuracy_score(y_test, y_pred)
        exactitudes.append(exactitud)
        K_list.append(k_vecinos)
    return K_list, exactitudes


def graficar_performance_kNN_kVariable(k_list, exactitudes, n_pixeles, medicion):
    plt.plot(k_list, exactitudes)
    plt.title(f'Performance de kNN según k - Distancia {medicion} - {n_pixeles} Pixeles')
    plt.xlabel('k vecinos')
    plt.ylabel('Exactitud')
    plt.ylim(0.95, 1)
    plt.show()



# ---- CLASIFICACIÓN MULTICLASE ---- #

## busco obtener la lista de los top 5 pixeles con mayor "variacion" de cada comparativa entre todas las clases.
def obtener_top_pixeles_varianza(datos_fashion, claseA, claseB, top_n=5):
    """
    Calcula los top_n píxeles con mayor varianza entre dos clases específicas.
    """
    # Filtramos los datos para obtener solo las filas de las clases deseadas
    datos_fashion_246 = duckdb.sql(f"""
        SELECT *
        FROM datos_fashion
        WHERE label = {claseA} OR label = {claseB}
    """).df()

    # Obtenemos solo las columnas de los píxeles
    pixeles = datos_fashion_246.loc[:, [col for col in datos_fashion.columns if col.startswith('pixel')]]

    # Obtenemos las etiquetas de clase
    labels = datos_fashion_246['label']

    # Calculamos el promedio por clase para cada píxel
    promedios_por_clase = pixeles.groupby(labels).mean()

    # Calculamos la varianza entre clases para cada píxel
    varianza_pixeles = promedios_por_clase.var()

    # Mostramos los top_n píxeles con mayor varianza entre clases
    top_pixeles = varianza_pixeles.sort_values(ascending=False).head(top_n).index.tolist()

    return top_pixeles

def obtener_pixeles_varianza_combinaciones(datos_fashion, top_n):
    """
    Recorre todas las combinaciones de clases, calcula los top_n píxeles con mayor varianza,
    y devuelve una lista de píxeles sin duplicados.
    """
    clases = range(10)  # Clases de 0 a 9
    top_pixeles_totales = set()  # Usamos un set para evitar duplicados

    # Generamos todas las combinaciones posibles de clases sin repetirse
    combinaciones_clases = itertools.combinations(clases, 2)

    # Para cada par de clases, obtenemos los píxeles con mayor varianza
    for claseA, claseB in combinaciones_clases:
        top_pixeles = obtener_top_pixeles_varianza(datos_fashion, claseA, claseB, top_n)
        top_pixeles_totales.update(top_pixeles)  # Agregamos los píxeles sin duplicar

    # Convertimos el set a lista y la devolvemos
    return list(top_pixeles_totales)



def performance_Decision_Tree(Profundidad, Variación, criterio):
    profundidades = [] # Guardo las distintas profundidades
    exactitudes = [] # Guardo las exactitudes para cada profundidad
    for profundidad in range (1, Profundidad + 1): # Me guardo una metrica para cada profundidad
        X_variaciones = X_dev[Variación]
        modelo_arbol = DecisionTreeClassifier(max_depth = profundidad, criterion = criterio) # Inicializo el arbol de decisión
        kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42) # Inicializo el kfold
        scores = cross_val_score(modelo_arbol, X_variaciones , Y_dev, cv = kfold, scoring = 'accuracy') # Guardo los scores (accuracys) del kfold
                                                                                         # Cada uno representa la accuracy obtenida con cada partición distinta
        profundidades.append(profundidad) # Guardo la profundidad con la que hice el modelo de cada iteración
        exactitudes.append(scores.mean()) # Guardo la media aritmetica de los scores obtenidos con kfold
    return profundidades, exactitudes
    # Repito con otra profundidad


def graficar_performance_Decision_Tree(profundidades, exactitudes, criterio, cant_atributos):
    plt.plot(profundidades, exactitudes)
    plt.title(f'Performance de profundidades según Profundidad - {criterio} - {cant_atributos} Píxeles')
    plt.xlabel('Profundidad')
    plt.ylabel('Exactitud')
    plt.show()



#%% EJECUCIÓN ANÁLISIS EXPLORATORIO

# Mapeo de las clases con su respectivo nombre
label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
    
graficar_cantidad_prendas_por_clase()

graficar_cantidad_instancias_con_info_por_atributo()

proyectar_valor_promedio_de_atributos([1,3])
proyectar_valor_promedio_de_atributos([0,2,4,6])
proyectar_valor_promedio_de_atributos([5,7,9])
proyectar_valor_promedio_de_atributos([8])



for i in range (10): # Se proyecta la imagen promedio de cada clase
    proyectar_imagen_promedio_de_clase(i)

    
proyectar_imagen_promedio_total()    
    
proyectar_varianza_clase(5)
proyectar_varianza_clase(0)



#%% EJECUCIÓN CLASIFICACIÓN BINARIA

varianza_matriz, pixeles_top_100 = n_pixeles_importantes(datos_fashion,100)
graficar_matriz_varianza(varianza_matriz)
pixeles_top_50 = n_pixeles_importantes(datos_fashion, 50)[1]
pixeles_top_150 = n_pixeles_importantes(datos_fashion,150)[1]


datos_fashion_0_8 = duckdb.sql(""" SELECT *
                                   FROM datos_fashion
                                   WHERE label = 0 OR label = 8""").df()



# Cantidad de cada prenda
print('Cantidad de cada prenda:', datos_fashion_0_8.label.value_counts())


X = datos_fashion_0_8.drop(columns = ['label'])
Y = datos_fashion_0_8.label


X = X[pixeles_top_100]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.1, random_state = 42, stratify = Y)

lista_k_euclidean_100Pixeles, exactitudes_euclidean_100Pixeles = performance_kNN_kVariable(30, 'euclidean', pixeles_top_100)
lista_k_euclidean_50Pixeles, exactitudes_euclidean_50Pixeles = performance_kNN_kVariable(30, 'euclidean', pixeles_top_50)
graficar_performance_kNN_kVariable(lista_k_euclidean_100Pixeles, exactitudes_euclidean_100Pixeles, 100, 'Euclidean')
graficar_performance_kNN_kVariable(lista_k_euclidean_50Pixeles, exactitudes_euclidean_50Pixeles, 50, 'Euclidean')
print("(Euclidean) Maxima Performance Alcanzada con 100 atributos:", max(exactitudes_euclidean_100Pixeles), "En k:", lista_k_euclidean_100Pixeles[exactitudes_euclidean_100Pixeles.index(max(exactitudes_euclidean_100Pixeles))])
print("(Euclidean) Maxima Performance Alcanzada con 50 atributos:", max(exactitudes_euclidean_50Pixeles), "En k:", lista_k_euclidean_50Pixeles[exactitudes_euclidean_50Pixeles.index(max(exactitudes_euclidean_50Pixeles))])

lista_k_manhattan_100Pixeles, exactitudes_manhattan_100Pixeles = performance_kNN_kVariable(30, 'manhattan', pixeles_top_100)
lista_k_manhattan_50Pixeles, exactitudes_manhattan_50Pixeles = performance_kNN_kVariable(30, 'manhattan', pixeles_top_50)
graficar_performance_kNN_kVariable(lista_k_manhattan_100Pixeles, exactitudes_manhattan_100Pixeles, 100, 'Manhattan')
graficar_performance_kNN_kVariable(lista_k_manhattan_50Pixeles, exactitudes_manhattan_50Pixeles, 50, 'Manhattan')
print("(Manhattan) Maxima Performance Alcanzada con 100 atributos:", max(exactitudes_manhattan_100Pixeles), "En k:", lista_k_manhattan_100Pixeles[exactitudes_manhattan_100Pixeles.index(max(exactitudes_manhattan_100Pixeles))])
print("(Manhattan) Maxima Performance Alcanzada con 50 atributos:", max(exactitudes_manhattan_50Pixeles), "En k:", lista_k_manhattan_50Pixeles[exactitudes_manhattan_50Pixeles.index(max(exactitudes_manhattan_50Pixeles))])


mejor_modelo_knn = KNeighborsClassifier(n_neighbors = 10, metric = 'euclidean')
mejor_modelo_knn.fit(x_train[pixeles_top_100], y_train)
prediccion_mejor_modelo_knn = mejor_modelo_knn.predict(x_test[pixeles_top_100])
print("T0, F8, F0, T8:", confusion_matrix(y_test, prediccion_mejor_modelo_knn).ravel()) # Toma el valor mas alto como positive, osea 8



#%% EJECUCIÓN CLASIFICACIÓN MULTICLASE

# Separar variables y etiquetas
X = datos_fashion.drop(columns=['label']) # Habría que usar los atributos 'útiles'
Y = datos_fashion['label']

# Dividir en desarrollo (90%) y validación held-out (10%)
X_dev, X_holdout, Y_dev, Y_holdout = train_test_split(X, Y, test_size=0.10, random_state=42, stratify=Y)
## uso la lista de top 5 pixeles con mayor variancia de cada comparativa clase a clase (se eliminan los repetidos)

# Obtener la lista de píxeles más variados entre todas las combinaciones de clases
top_pixeles_sin_repetir_5 = obtener_pixeles_varianza_combinaciones(datos_fashion, 5)
top_pixeles_sin_repetir_3 = obtener_pixeles_varianza_combinaciones(datos_fashion, 3)
top_pixeles_sin_repetir_1 = obtener_pixeles_varianza_combinaciones(datos_fashion, 1)

print(top_pixeles_sin_repetir_5)
print(top_pixeles_sin_repetir_3)
print(top_pixeles_sin_repetir_1)

cinco_pixeles_utiles = ['pixel686', 'pixel417', 'pixel427', 'pixel40', 'pixel597', 'pixel352', 'pixel400', 'pixel498', 'pixel372',
                  'pixel43', 'pixel606', 'pixel201', 'pixel596', 'pixel446', 'pixel97', 'pixel63', 'pixel582', 'pixel526',
                  'pixel76', 'pixel745', 'pixel510', 'pixel42', 'pixel389', 'pixel331', 'pixel353', 'pixel469', 'pixel146',
                  'pixel45', 'pixel455', 'pixel241', 'pixel373', 'pixel445', 'pixel94', 'pixel230', 'pixel173', 'pixel566',
                  'pixel162', 'pixel744', 'pixel38', 'pixel554', 'pixel502', 'pixel733', 'pixel231', 'pixel287', 'pixel69',
                  'pixel630', 'pixel594', 'pixel357', 'pixel473', 'pixel499', 'pixel658', 'pixel190', 'pixel271', 'pixel245',
                  'pixel380', 'pixel741', 'pixel602', 'pixel598', 'pixel441', 'pixel259', 'pixel244', 'pixel579', 'pixel242',
                  'pixel601', 'pixel343', 'pixel288', 'pixel527', 'pixel37', 'pixel434', 'pixel610', 'pixel743', 'pixel426',
                  'pixel303', 'pixel385', 'pixel98', 'pixel39', 'pixel359', 'pixel174', 'pixel398', 'pixel370', 'pixel70',
                  'pixel407', 'pixel90', 'pixel300', 'pixel638', 'pixel454', 'pixel472', 'pixel500', 'pixel71', 'pixel316',
                  'pixel399', 'pixel538', 'pixel482', 'pixel41', 'pixel740', 'pixel607', 'pixel371', 'pixel315', 'pixel202',
                  'pixel739', 'pixel528', 'pixel497', 'pixel605', 'pixel578', 'pixel260', 'pixel44', 'pixel600', 'pixel243',
                  'pixel218', 'pixel742', 'pixel418', 'pixel47', 'pixel599', 'pixel275', 'pixel246']

tres_pixeles_utiles = ['pixel638', 'pixel566', 'pixel41', 'pixel39', 'pixel434', 'pixel482', 'pixel426', 'pixel371', 'pixel739',
                       'pixel741', 'pixel398', 'pixel538', 'pixel389', 'pixel231', 'pixel599', 'pixel43', 'pixel526', 'pixel744',
                       'pixel740', 'pixel733', 'pixel331', 'pixel40', 'pixel372', 'pixel427', 'pixel601', 'pixel246', 'pixel98', 'pixel417',
                       'pixel658', 'pixel579', 'pixel527', 'pixel469', 'pixel202', 'pixel582', 'pixel602', 'pixel399', 'pixel610', 'pixel353',
                       'pixel173', 'pixel37', 'pixel303', 'pixel38', 'pixel287', 'pixel630', 'pixel370', 'pixel45', 'pixel242', 'pixel76', 'pixel44',
                       'pixel446', 'pixel288', 'pixel243', 'pixel230', 'pixel201', 'pixel500', 'pixel528', 'pixel69', 'pixel241', 'pixel316', 'pixel418',
                       'pixel445', 'pixel594', 'pixel554', 'pixel275', 'pixel497', 'pixel63', 'pixel598', 'pixel606', 'pixel259', 'pixel597', 'pixel343',
                       'pixel600', 'pixel743', 'pixel742', 'pixel42', 'pixel380', 'pixel510', 'pixel315', 'pixel174', 'pixel70', 'pixel218', 'pixel607']

pixel_mas_util = ['pixel40', 'pixel566', 'pixel554', 'pixel41', 'pixel497', 'pixel39', 'pixel63', 'pixel598', 'pixel606', 'pixel630', 'pixel259', 'pixel371',
                  'pixel246', 'pixel98', 'pixel417', 'pixel741', 'pixel242', 'pixel743', 'pixel446', 'pixel44', 'pixel528', 'pixel742', 'pixel380', 'pixel510',
                  'pixel398', 'pixel538', 'pixel202', 'pixel610', 'pixel399', 'pixel43', 'pixel173', 'pixel303', 'pixel38', 'pixel70']



X_dev = X_dev[cinco_pixeles_utiles]
X_holdout = X_holdout[cinco_pixeles_utiles]

profundidades_gini_115, exactitudes_gini_115 = performance_Decision_Tree(10, cinco_pixeles_utiles, 'gini')
profundidades_gini_69, exactitudes_gini_69 = performance_Decision_Tree(10, tres_pixeles_utiles, 'gini')
profundidades_gini_23, exactitudes_gini_23 = performance_Decision_Tree(10, pixel_mas_util, 'gini')

profundidades_entropy_115, exactitudes_entropy_115 = performance_Decision_Tree(10, cinco_pixeles_utiles, 'entropy')
profundidades_entropy_69, exactitudes_entropy_69 = performance_Decision_Tree(10, tres_pixeles_utiles, 'entropy')
profundidades_entropy_23, exactitudes_entropy_23 = performance_Decision_Tree(10, pixel_mas_util, 'entropy')

graficar_performance_Decision_Tree(profundidades_entropy_115, exactitudes_entropy_115, 'entropy', 115)
graficar_performance_Decision_Tree(profundidades_entropy_69, exactitudes_entropy_69, 'entropy', 69)
graficar_performance_Decision_Tree(profundidades_entropy_23, exactitudes_entropy_23, 'entropy', 23)

graficar_performance_Decision_Tree(profundidades_gini_115, exactitudes_gini_115, 'gini', 115)
graficar_performance_Decision_Tree(profundidades_gini_69, exactitudes_gini_69, 'gini', 69)
graficar_performance_Decision_Tree(profundidades_gini_23, exactitudes_gini_23, 'gini', 23)

# Encontrar la mejor profundidad y su exactitud

mejor_exactitud_gini_115 = max(exactitudes_gini_115)
mejor_profundidad_gini_115 = profundidades_gini_115[exactitudes_gini_115.index(mejor_exactitud_gini_115)]
print(f"Mejor exactitud gini: {mejor_exactitud_gini_115} - Mejor profundidad gini 115: {mejor_profundidad_gini_115}")

mejor_exactitud_gini_69 = max(exactitudes_gini_69)
mejor_profundidad_gini_69 = profundidades_gini_69[exactitudes_gini_69.index(mejor_exactitud_gini_69)]
print(f"Mejor exactitud gini: {mejor_exactitud_gini_69} - Mejor profundidad gini 69: {mejor_profundidad_gini_69}")

mejor_exactitud_gini_23 = max(exactitudes_gini_23)
mejor_profundidad_gini_23 = profundidades_gini_23[exactitudes_gini_23.index(mejor_exactitud_gini_23)]
print(f"Mejor exactitud gini: {mejor_exactitud_gini_23} - Mejor profundidad gini 23: {mejor_profundidad_gini_23}")

mejor_exactitud_entropy_115 = max(exactitudes_entropy_115)
mejor_profundidad_entropy_115 = profundidades_entropy_115[exactitudes_entropy_115.index(mejor_exactitud_entropy_115)]
print(f"Mejor exactitud entropy: {mejor_exactitud_entropy_115} - Mejor profundidad entropy 115: {mejor_profundidad_entropy_115}")

mejor_exactitud_entropy_69 = max(exactitudes_entropy_69)
mejor_profundidad_entropy_69 = profundidades_entropy_69[exactitudes_entropy_69.index(mejor_exactitud_entropy_69)]
print(f"Mejor exactitud entropy: {mejor_exactitud_entropy_69} - Mejor profundidad entropy 69: {mejor_profundidad_entropy_69}")

mejor_exactitud_entropy_23 = max(exactitudes_entropy_23)
mejor_profundidad_entropy_23 = profundidades_entropy_23[exactitudes_entropy_23.index(mejor_exactitud_entropy_23)]
print(f"Mejor exactitud entropy: {mejor_exactitud_entropy_23} - Mejor profundidad entropy 23: {mejor_profundidad_entropy_23}")


#Verificación de exactitud de modelo definitivo con HoldOut

modelo_definitivo = DecisionTreeClassifier(max_depth = 10, criterion = 'entropy')
modelo_definitivo.fit(X_dev[tres_pixeles_utiles], Y_dev)
Y_pred_definitivo = modelo_definitivo.predict(X_holdout[tres_pixeles_utiles])
print("Exactitud de Modelo Definitivo:", accuracy_score(Y_holdout, Y_pred_definitivo))