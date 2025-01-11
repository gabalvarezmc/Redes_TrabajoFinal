#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
#from Funciones.GEO_LIB import Plot, Poligono, Distancia, Geodecodificacion
from tqdm import tqdm
from scipy import stats
import numpy as np
import community as community_louvain
import networkx as nx
from itertools import count
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


df_base = pd.read_excel("./consolidado_reviews_Los_Rios_conINFO.xlsx").reset_index()
df_base = df_base.rename({'index':'Id'}, axis='columns')


df_base


print('Cantidad de Reviews: '+str(len(df_base)))
print('Cantidad de Comentarios: ' + str(len(df_base['review'].dropna())))
print('Cantidad de Lugares: '+ str(len(df_base['place_id'].unique())))
print('Cantidad de usuarios únicos: '+str(len(df_base['id_usuario'].unique())))


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


W = nx.Graph()
for row in list_cooc_2.itertuples():
    W.add_edge(row.i, row.j, weight=row.pairs)


nx.write_gexf(W, "Red_sin_Filtrar.gexf")


dict10=dict(W.degree())  # node 0 has degree 1
sorted_dict0 = {}
sorted_keys0 = sorted(dict10, key=dict10.get)  # [1, 3, 2]

for w in sorted_keys0:
    sorted_dict0[w] = dict10[w]
sorted_dict0


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


# Esta sección hace el análisis de la matriz de co-ocurrencia.
# 
# Primero transforma el DataFrame `list_cooc` en un formato de matriz, donde las filas y columnas son los distintos elementos y los valores de las celdas indican la frecuencia de co-ocurrencia. Los elementos diagonales de esta matriz, que representan auto-enlaces, se tiran a 0, por que los deja fuera del análisis.
# 
# Segundo se crean varias matrices vacias (`phi`, `tval`, `pval`, `phi2` y `exit`) para almacenar distintas medidas estadísticas. La matriz `phi` para los coeficientes de correlación phi, `tval` para los t-values, `pval` para los valores p (la significancia), `phi2` almacenará los pesos finales y `exit` es una matriz binaria que indica si una correlación específica es significativa basada en `pval_limit` y `cnt_limit`.
# 
# Tercero entra en un bucle anidado para calcular los coeficientes de correlación phi para cada par de elementos. El coeficiente phi se calcula utilizando considerando el número total de observaciones y las frecuencias de los elementos. Si el denominador de esta fórmula es cero, el coeficiente phi se guarda como cero para evitar errores de división por cero. Luego se calculan los valores t y los valores p para evaluar la significancia de estas correlaciones. Si la correlación cumple con los criterios de significancia ( `pval` menor que `pval_limit` y coocurencias mayor a `cnt_limit`), se marca como 1 en la matriz `exit` y su coeficiente phi se almacena en `phi2`.
# 
# Después de calcular las medidas estadísticas, el código construye DataFrames para almacenar los coeficientes phi y los valores p. Estos DataFrames se combinan con los datos originales de co-ocurrencia para crear un DataFrame final que incluye el coeficiente phi, el valor p y el número de veces que cada par de elementos co-ocurrió.
# 
# Finalmente, el código visualiza el número de nodos y aristas en la red en función del umbral de valores p. Genera un rango de valores p y filtra el DataFrame final para incluir solo aquellos pares con valores p por debajo de cada umbral. Luego se calcula y se grafica el número de nodos únicos y aristas para mostrar cómo cambia la estructura de la red con diferentes niveles de significancia. Los gráficos se muestran con líneas verticales que indican umbrales específicos de valores p como referencia.
# 

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



# Filtered
F = S[0]
dict1=dict(F.degree())  # node 0 has degree 1
sorted_dict = {}
sorted_keys = sorted(dict1, key=dict1.get)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = dict1[w]

#plt.hist((dict1.values()), bins=np.logspace(0,3))
#plt.loglog()
plt.hist((dict1.values()), bins=50)
plt.show()


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
#https://datoslab.cl/hes/

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

