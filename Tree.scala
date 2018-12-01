import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType};
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import scala.io.Source
import java.io._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LinearSVC
import org.apache.log4j._


Logger.getLogger("org").setLevel(Level.ERROR)
//Crear una simple sesion es spark.
val spark = SparkSession.builder().getOrCreate()
//cargar el Dataset.
val df = spark.read.option("header", true).option("inferSchema", "true").option("delimiter", ";").csv("bank-full.csv")

//exploracion de los datos
df.printSchema
df.show(3)
df.head
df.columns
//Reemplazar los datos de tipo String a numéricos.

val df2 = df.withColumn("label", when(col("y") === "yes", 1).otherwise(2))

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


val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous", "InAge", "InJob", "InMarital", "InEducation",
"InDefault", "InHousing", "InLoan", "InPdays", "InPrevious", "InPoutcome")).setOutputCol("caracteristicas")
val output = assembler.transform(indexer9)
val data = output.select("label","caracteristicas")
data.show(50,false)
data.na.drop()

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
val normalizer = new Normalizer().setInputCol("caracteristicas").setOutputCol("features").setP(1.0)

val l1NormData = normalizer.transform(data)
println("Normalized using L^1 norm")
l1NormData.show(50)

val df = l1NormData.select("label","features")
df.show(false)

val splits = df.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
//////terminacion de limpieza de datos/////////////////////////


import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)// features with > 4 distinct values are treated as continuous.  .fit(data)
// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))
// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)
// Make predictions.
val predictions = model.transform(testData)
// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
