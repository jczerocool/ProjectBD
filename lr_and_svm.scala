//PRIMERA PARTE.
//Limpieza de datos
//inicio de session en spark apache
//librerias necesarias
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType};
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoderEstimator}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.Pipeline
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
//carga de dataset
val df = spark.read.option("header", true).option("inferSchema", "true").option("delimiter", ";").csv("bank-full.csv")
//creacion de una nueva tabla que contendra la columna "y", la cualcontiene string y/n, para convertirla
//en numerica 1.0  y 0.0
//1.0 cuando sea yes y 0.0 cuando es no, esto es posible gracias a(when, .otherwise )
val df2 = when($"y".contains("yes"), 1.0).otherwise(0.0)
//se creara una nueva tabla en la cual estaran contenidas la tabla del paso anterior (df2)
// mas la tabla original
//df3 contiene los datos de y en formato numerico
val df3 = df.withColumn("y", df2)
//despliege de los datos que contiene df3
df3.show(2)
//esquema de la talba df3
df3.printSchema
df3.head
df3.columns

// Convertir strings a valores numericos
//se toman los campos del dataset que es necesario convertir de string a numerico
//esto con la finalidad de tomar las caracteristicas que ayuden a incrementar
//la precision
//se crean variables por cada uno de las columnas
//hacemos uso de StringIndexer para lograr el cambio
//para no crear una nueva tabla contenedor para estas columnas
// se utiliza .fit(df3), la funcion de esta es reemplazar la columna
// vieja por una esta nueva
val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex").fit(df3)
val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex").fit(df3)
val eduIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex").fit(df3)
val defaultIndexer = new StringIndexer().setInputCol("default").setOutputCol("defaultIndex").fit(df3)
val housingIndexer = new StringIndexer().setInputCol("housing").setOutputCol("housingIndex").fit(df3)
val loanIndexer = new StringIndexer().setInputCol("loan").setOutputCol("loanIndex").fit(df3)
val contactIndexer = new StringIndexer().setInputCol("contact").setOutputCol("contactIndex").fit(df3)
val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex").fit(df3)

//los Index obtenidos en el paso anterior se transformaran  y su resultado
//reemplezara los datos en nuestra tabla

val jobIndexed = jobIndexer.transform(df3)
val maritalIndexed = maritalIndexer.transform(df3)
val eduIndexed = eduIndexer.transform(df3)
val defaultIndexed = defaultIndexer.transform(df3)
val housingIndexed = housingIndexer.transform(df3)
val loanIndexed = loanIndexer.transform(df3)
val contactIndexed = contactIndexer.transform(df3)
val monthIndexed = monthIndexer.transform(df3)

//se crea un nuevo Encoder el cual contendra las columnas indexadas en el paso anterior
// y con estas crera vecrores
val Encoder = new OneHotEncoderEstimator().setInputCols(Array("jobIndex", "maritalIndex", "educationIndex", "defaultIndex", "housingIndex", "loanIndex", "contactIndex", "monthIndex")).setOutputCols(Array("jobVec", "maritalVec", "educationVec", "defaultVec", "housingVec", "loanVec", "contactVec", "monthVec"))
//assembler toma los datos que se desea utilizar de la tabla y los convierte
//en un vector con "features", esta nueva columna contendra todos los datos de
//las columas seleccionadas.
val assembler = (new VectorAssembler().setInputCols(Array("age","duration", "balance","day","campaign", "previous", "jobVec", "maritalVec", "educationVec", "defaultVec", "housingVec", "loanVec", "contactVec", "monthVec")).setOutputCol("features"))
//dividimos el dataset en 70% para entrenamiento y 30% para pruebas
//ademas de asgnar una semilla 12345, ela funcion de esta que los usuarios
//que utilizen este algoritmo puedan obtener los mismos resultados.
val Array(training, test) = df3.randomSplit(Array(0.7, 0.3), seed = 12345)

//ESTO CONCLUYE LA PARTE DE LA LIMPIEZA DE DATOS.
///////////////////////LogisticRegression///////////////////////////////////
//IMPORTAMOS import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegression
// creamos una set de entrenamiento y tomamos la columas y
// de la cual nos interesa saber el porcentaje de que el usaurio suscribto
//reciba un deposito, se le dara un maximo de 10 iteraciones
//para evitar que se cicle.
val lr = new LogisticRegression().setLabelCol("y").setMaxIter(10)
//pipeline toma los datos de Indexer, enconder y assemmbleer
val pipeline_lr = new Pipeline().setStages(Array(jobIndexer,maritalIndexer, eduIndexer, defaultIndexer, housingIndexer, loanIndexer, contactIndexer, monthIndexer, Encoder, assembler,lr))
//el modelo contiene los datos asigandos en pipeline y los entrena
val model_lr = pipeline_lr.fit(training)//error con las columnas que son strings
//una vez entrenados los datos se procede a ralizar las pruebas
val results_lr = model_lr.transform(test)
//predictionAndLabels contendra los resultados obtenidos en el paso anterior
//de este se seleccionan la $"prediction",$"y" como datos dobles
val predictionAndLabels_lr = results_lr.select($"prediction",$"y").as[(Double, Double)].rdd
//metrics usara metrics_lr
val metrics_lr = new MulticlassMetrics(predictionAndLabels_lr)
//println("Confusion matrix:")
println(metrics_lr.confusionMatrix)
//println("accuracy")
println(metrics_lr.accuracy)

/////////////////SVM//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
import org.apache.spark.ml.classification.LinearSVC

// Load training data ya no es necessario cargarlo puesto que ya se hizo en la fase anterior
//val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
val lsvc = new LinearSVC().setLabelCol("y").setMaxIter(10)//.setRegParam(0.1)
//val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
val pipeline_lsvc = new Pipeline().setStages(Array(jobIndexer,maritalIndexer, eduIndexer, defaultIndexer, housingIndexer, loanIndexer, contactIndexer, monthIndexer, Encoder, assembler, lsvc))

val model_lsvc = pipeline_lsvc.fit(training)//error con las columnas que son strings

val results_lsvc = model_lsvc.transform(test)

val predictionAndLabels_lsvc = results_lsvc.select($"prediction",$"y").as[(Double, Double)].rdd

val metrics_lsvc = new MulticlassMetrics(predictionAndLabels_lsvc)

//println("Confusion matrix:")
println(metrics_lsvc.confusionMatrix)
metrics_lsvc.accuracy
