%pyspark
import geomesa_pyspark
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import sqlite3
import itertools

LO_ARRAY = [32.5624510, 32.563501, 32.550874, 32.5299,32.632199,3.471984,3.420471,3.4053752,3.3976159,3.3765,3.2888,3.3751,3.312947,3.325671,-10.77351,-10.806985,-10.73743,-10.814004,-10.816781,]
LA_ARRAY = [15.529662,15.531861,15.572177,15.6051,15.617421,6.4322,6.432992,6.4449293,6.4492307,6.4605,6.4618,6.5198,6.531487,6.575957,6.286724,6.308429,6.309258,6.312883,6.315304,6.3697076]

#LOADED_MODEL = load_model("model.h5")

conf = geomesa_pyspark.configure(
    jars=['/opt/depZep/geomesa-accumulo-spark-runtime_2.11-2.0.0-m.1.jar'],
    packages=['geomesa_pyspark','pytz'],
    spark_home='/usr/hdp/current/spark2-client').\
    setAppName('PySparkTestApp')

conf.get('spark.master')

from pyspark.sql import SparkSession

spark = ( SparkSession
    .builder
    .config(conf=conf)
    .enableHiveSupport()
    .getOrCreate()
)

params = {
    "instanceId": "hdp-accumulo-instance",
    "zookeepers": "hdp-cluster-node3.symphony.org:2181,hdp-cluster-node8.symphony.org:2181,hdp-cluster-node2.symphony.org:2181",
    "user": "student",
    "password": "student",
    "tableName": "geomesa.population"
}

feature = "Africa"
postDf = ( spark
    .read
    .format("geomesa")
    .options(**params)
    .option("geomesa.feature", feature)
    .load()
)

postDf.createOrReplaceTempView("africa")
postDf.registerTempTable("africa")
sqlContext.cacheTable("africa")
#spark.sql("show tables").show()


query1 = spark.sql("""
select gridcode, Shape_Leng, Shape_Area
from africa
where country="Liberia" OR country="Nigeria" OR country="Sudan" AND
(st_contains(st_makeBBOX(-10.7015803,6.230721,-10.685803,6.250721), the_geom) OR
st_contains(st_makeBBOX(-10.7000157,6.2734063,-10.6800157,6.2934063), the_geom) OR
st_contains(st_makeBBOX(-10.765022,6.263372,-10.745022,6.283372), the_geom) OR
st_contains(st_makeBBOX(-10.819247,6.309735,-10.79247,6.329735), the_geom) OR
st_contains(st_makeBBOX(-10.790388,6.286402,-10.770388,6.306402), the_geom) OR
st_contains(st_makeBBOX(3.14341,6.454712,3.16341,6.474712), the_geom) OR
st_contains(st_makeBBOX(3.2412224,6.5520264,3.2612224,6.5720264), the_geom) OR
st_contains(st_makeBBOX(3.364938,6.500547,3.384938,6.520547), the_geom) OR
st_contains(st_makeBBOX(3.338812,6.431817,3.378812,6.45817), the_geom) OR
st_contains(st_makeBBOX(3.359724,6.455453,3.399724,6.47453), the_geom) OR
st_contains(st_makeBBOX(32.534326,15.663091,32.554326,15.683091), the_geom) OR
st_contains(st_makeBBOX(32.533284,15.567043,32.553284,15.587043), the_geom) OR
st_contains(st_makeBBOX(32.577521,15.562588,32.583284,15.582588), the_geom) OR
st_contains(st_makeBBOX(32.538319,15.539017,32.558319,15.559017), the_geom) OR
st_contains(st_makeBBOX(32.561023,15.589514,32.581023,15.609514), the_geom))""").limit(1000)

pandaDF1=query1.toPandas()
positives=pandaDF1.values
positives= np.insert(positives, 3, 1, axis=1)


query2 = spark.sql("""
select gridcode, Shape_Leng, Shape_Area
from africa
where country="Liberia" OR country="Nigeria" OR country="Sudan" AND not
(st_contains(st_makeBBOX(-10.7015803,6.230721,-10.685803,6.250721), the_geom) OR
st_contains(st_makeBBOX(-10.7000157,6.2734063,-10.6800157,6.2934063), the_geom) OR
st_contains(st_makeBBOX(-10.765022,6.263372,-10.745022,6.283372), the_geom) OR
st_contains(st_makeBBOX(-10.819247,6.309735,-10.79247,6.329735), the_geom) OR
st_contains(st_makeBBOX(-10.790388,6.286402,-10.770388,6.306402), the_geom) OR
st_contains(st_makeBBOX(3.14341,6.454712,3.16341,6.474712), the_geom) OR
st_contains(st_makeBBOX(3.2412224,6.5520264,3.2612224,6.5720264), the_geom) OR
st_contains(st_makeBBOX(3.364938,6.500547,3.384938,6.520547), the_geom) OR
st_contains(st_makeBBOX(3.338812,6.431817,3.378812,6.45817), the_geom) OR
st_contains(st_makeBBOX(3.359724,6.455453,3.399724,6.47453), the_geom) OR
st_contains(st_makeBBOX(32.534326,15.663091,32.554326,15.683091), the_geom) OR
st_contains(st_makeBBOX(32.533284,15.567043,32.553284,15.587043), the_geom) OR
st_contains(st_makeBBOX(32.577521,15.562588,32.583284,15.582588), the_geom) OR
st_contains(st_makeBBOX(32.538319,15.539017,32.558319,15.559017), the_geom) OR
st_contains(st_makeBBOX(32.561023,15.589514,32.581023,15.609514), the_geom))""").limit(1000)

pandaDF2=query2.toPandas()
negatives=pandaDF2.values
negatives= np.insert(negatives, 3, 0, axis=1)

train = np.concatenate((positives, negatives), axis=0)
print(train)

x_train = train[:,0:-1]
y_train = train[:,-1]

model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=3))
model.add(Dropout(0.5))
'''model.add(Dense(units=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=20, activation='relu'))
model.add(Dropout(0.5))'''
model.add(Dense(units=10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='acc', min_delta=0, verbose=0, mode='auto')
model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[earlyStopping])

#model.save("model.h5")
for i in range(20):

    query3 = spark.sql("""
select gridcode, Shape_Leng, Shape_Area FROM africa WHERE  (country="Liberia" OR country="Sudan" OR country="Nigeria" AND
(st_contains(st_makeBBOX(""" + str(LATITUDE - .01) + """,""" + str(LONGITUDE - .01) + """,""" + str(LATITUDE + .01) +""",""" + str(LONGITUDE + .01)+"""), the_geom)
))""").limit(1)

    pandaDF3=query3.toPandas()
    test=pandaDF3.values
    x_test = np.concatenate((x_test, test), axis=0)

classes = model.predict(x_test, batch_size=128)
print(classes)
