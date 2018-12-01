import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType};

//Prevenir errores.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Crear una simple sesion es spark.
val spark = SparkSession.builder().getOrCreate()


//Leer el Dataset.
val df = spark.read.option("header", true).option("inferSchema", "true").option("delimiter", ";").csv("bank-full.csv")

//Ver Estructura de los datos.
df.printSchema
df.show(3)
df.head
df.columns

//Eliminar posibles solumnas con datos nulos.
val data = df.na.drop()

//Reemplazar los datos de tipo String a numéricos.
import org.apache.spark.ml.feature.StringIndexer

val indexer_age = new StringIndexer().setInputCol("age").setOutputCol("AgeIndex")
val indexer_job = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
val indexer_marital = new StringIndexer().setInputCol("marital").setOutputCol("MaritalIndex")
val indexer_education = new StringIndexer().setInputCol("education").setOutputCol("EducationIndex")
val indexer_default = new StringIndexer().setInputCol("default").setOutputCol("DefaultIndex")
val indexer_balance = new StringIndexer().setInputCol("balance").setOutputCol("BalanceIndex")
val indexer_housing = new StringIndexer().setInputCol("housing").setOutputCol("HousingIndex")
val indexer_loan = new StringIndexer().setInputCol("loan").setOutputCol("LoanIndex")
val indexer_contact = new StringIndexer().setInputCol("contact").setOutputCol("ContactIndex")
val indexer_day = new StringIndexer().setInputCol("day").setOutputCol("DayIndex")
val indexer_month = new StringIndexer().setInputCol("month").setOutputCol("MonthIndex")
val indexer_duration = new StringIndexer().setInputCol("duration").setOutputCol("DurationIndex")
val indexer_campaign = new StringIndexer().setInputCol("campaign").setOutputCol("CampaignIndex")
val indexer_pdays = new StringIndexer().setInputCol("pdays").setOutputCol("PdaysIndex")
val indexer_previous = new StringIndexer().setInputCol("previous").setOutputCol("PreviousIndex")
val indexer_poutcome = new StringIndexer().setInputCol("poutcome").setOutputCol("PoutcomeIndex")
val indexer_y = new StringIndexer().setInputCol("y").setOutputCol("label")

val indexed = indexer_job.fit(data).transform(data)
val data2 = indexed
val indexed = indexer_marital.fit(data2).transform(data2)
val data3 = indexed
val indexed = indexer_education.fit(data3).transform(data3)
val data4 = indexed
val indexed = indexer_default.fit(data4).transform(data4)
val data5 = indexed
val indexed = indexer_housing.fit(data5).transform(data5)
val data6 = indexed
val indexed = indexer_loan.fit(data6).transform(data6)
val data7 = indexed
val indexed = indexer_contact.fit(data7).transform(data7)
val data8 = indexed
val indexed = indexer_month.fit(data8).transform(data8)
val data9 = indexed
val indexed = indexer_poutcome.fit(data9).transform(data9)
val data10 = indexed
val indexed = indexer_y.fit(data10).transform(data10)
val data11 = indexed
val indexed = indexer_age.fit(data11).transform(data11)
val data12 = indexed
val indexed = indexer_balance.fit(data12).transform(data12)
val data13 = indexed
val indexed = indexer_day.fit(data13).transform(data13)
val data14 = indexed
val indexed = indexer_duration.fit(data14).transform(data14)
val data15 = indexed
val indexed = indexer_campaign.fit(data15).transform(data15)
val data16 = indexed
val indexed = indexer_pdays.fit(data16).transform(data16)
val data17 = indexed
val indexed = indexer_previous.fit(data17).transform(data17)

val df2 = indexed
df2.show()

val df3 = df2.select($"label", $"AgeIndex", $"JobIndex", $"MaritalIndex", $"EducationIndex", $"DefaultIndex", $"BalanceIndex", $"HousingIndex",
$"LoanIndex", $"ContactIndex", $"DayIndex", $"MonthIndex", $"DurationIndex", $"CampaignIndex", $"PdaysIndex", $"PreviousIndex", $"PoutcomeIndex")

///////////////////hasta obtenemos la tabla con datos numericos ////////////////////////
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("AgeIndex", "JobIndex", "MaritalIndex", "EducationIndex", "DefaultIndex", "BalanceIndex", "HousingIndex",
"LoanIndex", "ContactIndex", "DayIndex", "MonthIndex", "DurationIndex", "CampaignIndex", "PdaysIndex", "PreviousIndex", "PoutcomeIndex")).setOutputCol("features")

val output = assembler.transform(df3)

output.select("label", "features").show(false)


val mlp = output.select("label", "features")
mlp.show(false)
mlp.printSchema


import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row

// Load training data
//val data = MLUtils.loadLibSVMFile(sc, "bank-full.csv").toDF()
//val mlpc = new MLUtils().setLabelCol("y").setMaxIter(10).setRegParam(0.1)


// Load training data
//val data = output
//data.show(2)
// Load training data
// Split the data into train and test
val splits = mlp.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
val layers = Array[Int](16, 2, 2, 5)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// train the model
val model = trainer.fit(train)

val result = model.transform(test)

val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
