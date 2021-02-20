#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[74]:


retail = pd.read_csv('frfr.csv')
retail.head()


# In[75]:


# df info

retail.info()


# In[76]:


# df description

retail.describe()


# ## Шаг 2 : Очистка данных
# Расчет недостающих значений в процентном соотношении  в DF

# In[77]:




df_null = round(100*(retail.isnull().sum())/len(retail), 2)
df_null


# In[78]:


# Очистка от строк с пропущенными значениями 

retail = retail.dropna()
retail.shape


# In[79]:


# Изменение типа данных для ID Клиента 
retail['CustomerID'] = retail['CustomerID'].astype(str)


# ## Шаг 3: Подготовка данных
# 
# Мы собираемся анализировать клиентов на основе следующих трех факторов:
# - R (Recency) количество дней с момента последней покупки клиента.
# - F (Frequency): количество заказов клиента.
# - M (Monetary): общая сумма транзакций клента.

# In[80]:




# Добавим новый атрибут : Monetary

# retail['Amount'] = retail['cash_price']
rfm_m = retail.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()


# In[81]:


# Добавим новый атрибут : Frequency

rfm_f = retail.groupby('CustomerID')['OrderID'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()


# In[82]:


# Объединим две таблиы данных

rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm.head()


# In[83]:


# Добавим новый атрибут : Recency

# Сделаем конвертацию даты в нужный формат для дальнейшей работы

retail['Created_at'] = pd.to_datetime(retail['Created_at'],format='%Y-%m-%d %H:%M')


# In[84]:


# Вычислим время самой последней транзакции для каждого клиента

max_date = max(retail['Created_at'])
max_date


# In[85]:


# Вычислим для всех заказов разницу между временем создания заказа и временем создания последнего заказа

retail['Diff'] = max_date - retail['Created_at']
retail.head()


# In[86]:


#  Вычислим Recency, "давность оформления заказа" для каждого клиента

rfm_p = retail.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()


# In[87]:


# Посчитаем в днях

rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()


# In[88]:


# Объединим таблицы в целое, получим RFM таблицу
rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()


# ### Есть 2 типа выбросов, которые мы будем обрабатывать:
# - Статистический
# - Зависящий от домена
# 

# In[89]:



# Анализ выбросов

attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')


# In[90]:


# Удаление выбросов для Amount
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Удаление выбросов Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Удаление выбросов для Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]


# ### Изменение масштаба атрибутов
#  важно изменить масштаб переменных, чтобы они имели сопоставимый масштаб. Есть два распространенных способа изменения масштаба:
# 
# - Минимакс масштабирование
# - Стандартизация (среднее-0, сигма-1)
# Будем использовать стандартизацию.
# Цель — преобразовать исходный набор в новый со средним значением равным 0 и стандартным отклонением равным 1.

# In[91]:




# Изменение масштаба

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]


scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[ ]:





# In[20]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()


# In[117]:



#fig, axes = plt.subplot(1,3, figsize=(20,5))
#for i, feature in enumerate(list(rfm.columns)):
sns.displot(rfm_df.Amount)
sns.displot(rfm_df.Frequency)


# In[93]:





# # Шаг 4: Создание моделей 
# # K-Means кластеризация
# 
# Алгоритм работает следующим образом:
# Сначала мы случайным образом выбираем k точек
# Мы классифицируем каждый элемент по ближайшему среднему значению и обновляем координаты среднего значения, которые представляют собой средние значения элементов, отнесенных к этому среднему значению на данный момент.
# Мы повторяем процесс для заданного количества итераций, и в конце получаем кластеры

# In[110]:


# k-means 

kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[118]:


## k=3 визуалиация
km3=KMeans(n_clusters=3,init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km4.fit_predict(rfm_df_scaled)
plt.scatter(rfm_df_scaled[y_means==0,0],rfm_df_scaled[y_means==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(rfm_df_scaled[y_means==1,0],rfm_df_scaled[y_means==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(rfm_df_scaled[y_means==2,0],rfm_df_scaled[y_means==2,1],s=50, c='green',label='Cluster3')
#plt.scatter(rfm_df_scaled[y_means==3,0],rfm_df_scaled[y_means==3,1],s=50, c='cyan',label='Cluster4')
plt.scatter(km4.cluster_centers_[:,0], km4.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('Customer segments')
plt.xlabel('Frequency')
plt.ylabel('Amounr')
plt.legend()
plt.show()


# In[22]:


kmeans.labels_


# In[23]:


# Нахождение оптимального количества кластеров

# Метод локтя является эвристическим используемым в определении количества кластеров в наборе данных.
# Метод состоит из построения объясненного изменения как функции количества кластеров и выбора изгиба кривой
# в качестве количества используемых кластеров

# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)


# ### Silhouette Analysis
# 
# $$\text{silhouette score}=\frac{p-q}{max(p,q)}$$
# * Силуэт относится к методу интерпретации и проверки согласованности в пределах кластеров данных . Этот метод дает краткое графическое представление о том, насколько хорошо каждый объект был классифицирован. Значение силуэта является мерой того, насколько объект похож на его собственный кластер (сцепление) по сравнению с другими кластерами (разделение). Силуэт находится в диапазоне от -1 до +1, где высокое значение указывает, что объект хорошо соответствует своему собственному кластеру и плохо соответствует соседним кластерам. Если большинство объектов имеют высокое значение, то конфигурация кластеризации подходит. Если многие точки имеют низкое или отрицательное значение, то в конфигурации кластеризации может быть слишком много или слишком мало кластеров. Силуэт можно рассчитать с помощью любой метрики расстояния , такой как евклидово расстояние или манхэттенское расстояние

# In[149]:


## Silhouette analysis
# https://gdcoder.com/silhouette-analysis-vs-elbow-method-vs-davies-bouldin-index-selecting-the-optimal-number-of-clusters-for-kmeans-clustering/

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    


# In[24]:


# модель с  k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[25]:


kmeans.labels_


# In[26]:



rfm['Cluster_Id'] = kmeans.labels_
rfm.head()


# In[27]:


# Box plot для визуализации Cluster Id vs Amount

sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)


# In[28]:


# Box plot для визуализации Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)


# In[29]:


# Box plot для визуализации Cluster Id vs Recency

sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)


# In[41]:



#filter rows of original data cluster_labels
label = kmeans.fit(rfm_df_scaled)
filtered_label0 = rfm[ label == 0]

#plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()
plt.show()


# ### Иерархическая кластеризация
# 
# Иерархическая кластеризация включает создание кластеров, которые имеют заранее определенный порядок сверху вниз. Например, все файлы и папки на жестком диске организованы в иерархию. Есть два типа иерархической кластеризации:
# - Разделительный
# - Агломеративный

# **Single Linkage:<br>**
# 
# 1.	Одиночная связь (расстояния ближайшего соседа)
# В этом методе расстояние между двумя кластерами определяется расстоянием между двумя наиболее близкими объектами (ближайшими соседями) в различных кластерах. Результирующие кластеры имеют тенденцию объединяться в цепочки.
# 
# 
# 
# ![](https://www.saedsayad.com/images/Clustering_single.png)

# In[30]:


## Single linkage: 

mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()


# **Complete Linkage<br>**
# 
# 2.	Полная связь (расстояние наиболее удаленных соседей)
# В этом методе расстояния между кластерами определяются наибольшим расстоянием между любыми двумя объектами в различных кластерах (т.е. наиболее удаленными соседями). Этот метод обычно работает очень хорошо, когда объекты происходят из отдельных групп. Если же кластеры имеют удлиненную форму или их естественный тип является «цепочечным», то этот метод непригоден.
# 
# 
# 
# ![](https://www.saedsayad.com/images/Clustering_complete.png)

# In[62]:


# Complete linkage

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# **Average Linkage:<br>**
# 
# Метод средней связи (average linkage) действует аналогично. Однако в этом методе рас­стояние между двумя кластерами определяют как среднее значение всех расстояний, изме->енных между объектами двух кластеров, при этом в каждую пару входят объекты из разных сластеров  в различных кластерах. Результирующие кластеры имеют тенденцию объединяться в цепочки.
# nt in one cluster to every point in the other cluster. For example, the distance between clusters “r” and “s” to the left is equal to the average length each arrow between connecting the points of one cluster to the other.
# ![](https://www.saedsayad.com/images/Clustering_average.png)
# 

# In[ ]:


# Average linkage


mergings = linkage(rfm_df_scaled, method="average", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[29]:


# Cutting the Dendrogram based on K
# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


# In[64]:




rfm['Cluster_Labels'] = cluster_labels
rfm.head()


# In[65]:


# Plot Cluster Id vs Amount

sns.boxplot(x='Cluster_Labels', y='Amount', data=rfm)


# In[66]:


# Plot Cluster Id vs Frequency

sns.boxplot(x='Cluster_Labels', y='Frequency', data=rfm)


# In[67]:


# Plot Cluster Id vs Recency

sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm)


# Тут будет вывод 
# 

# In[70]:



rfm.to_csv(writer,'AR')
writer.save()


# In[ ]:




