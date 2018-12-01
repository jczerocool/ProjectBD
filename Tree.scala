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
val dt = spark.read.option("header", true).option("inferSchema", "true").option("delimiter", ";").csv("bank-full.csv")

//exploracion de los datos
dt.printSchema
dt.show(3)
dt.head
dt.columns
//Reemplazar los datos de tipo String a numéricos.

val df2 = dt.withColumn("label", when(col("y") === "yes", 1).otherwise(2))

val indexer = new StringIndexer().setInputCol("age").setOutputCol("InAge").fit(df2).transform(df2)
val indexer1 = new StringIndexer().setInputCol("job").setOutputCol("InJob").fit(indexer).transform(indexer)
val indexer2 = new StringIndexer().setInputCol("marital").setOutputCol("InMarital").fit(indexer1).transform(indexer1)
val indexer3 = new StringIndexer().setInputCol("education").setOutputCol("InEducation").fit(indexer2).transform(indexer2)
val indexer4 = new StringIndexer().setInputCol("default").setOutputCol("InDefault").fit(indexer3).transform(indexer3)
val indexer5 = new StringIndexer().setInputCol("housing").setOutputCol("InHousing").fit(indexer4).transform(indexer4)
val indexer6 = new StringIndexer().setInputCol("loan").setOutputCol("InLoan").fit(indexer5).transform(indexer5)
val indexer7= new StringIndexer().setInputCol("pdays").setOutputCol("InPdays").fit(indexer6).transform(indexer6)
val indexer8 = new StringIndexer().setInputCol("previous").setOutputCol("InPrevious").fit(indexer7).transform(indexer7)
val indexer9 = new StringIndexer().setInputCol("poutcome").setOutputCol("InPoutcome").fit(indexer8).transform(indexer8)

val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous", "InAge", "InJob", "InMarital", "InEducation",
"InDefault", "InHousing", "InLoan", "InPdays", "InPrevious", "InPoutcome")).setOutputCol("caracteristicas")
val output = assembler.transform(indexer9)
val data = output.select("label","caracteristicas")
data.show(50,false)
data.na.drop()

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
val normalizer = new Normalizer()
.setInputCol("caracteristicas")
.setOutputCol("features")
.setP(1.0)

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
