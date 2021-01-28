package xyz.stuffium

import com.typesafe.scalalogging.LazyLogging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import xyz.stuffium.util.Importer

object TP2 extends LazyLogging {

  org.slf4j.LoggerFactory
    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
    .asInstanceOf[ch.qos.logback.classic.Logger]
    .setLevel(ch.qos.logback.classic.Level.INFO)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("xyz").setLevel(Level.WARN)
    logger.info("Hewos")

    val spark = SparkSession
      .builder()
      .appName("ri-tp2")
      .master("local[*]")
      .getOrCreate()

    val trainRaw = loadData(spark)

    val indexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")

    val sim = indexer.fit(trainRaw)
    val train = sim.transform(trainRaw)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("tokensSW")

    val remover = new StopWordsRemover()
      .setLocale("en_US")
      .setInputCol("tokensSW")
      .setOutputCol("tokens")

    val hashingTF = new HashingTF()
      .setInputCol("tokens")
      .setOutputCol("rawTF")

    val idf = new IDF()
      .setInputCol("rawTF")
      .setOutputCol("tf-idf")
      .setMinDocFreq(2)

    val chiSq = new ChiSqSelector()
      .setFeaturesCol("tf-idf")
      .setLabelCol("label")
      .setOutputCol("features")

    val nb = new NaiveBayes()

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, chiSq, nb))

    logger.warn("pipeline done")

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100, 200, 500, 1000))
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200, 500))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    logger.warn("Gonna create model")

//    val model = cv.fit(train)
//    println(model.avgMetrics.toList)
//    model.write.overwrite.save("./models/nb_cv")
    val model = CrossValidatorModel.load("./models/nb_cv")

    val r = model.transform(train)
    println(model.bestModel.params.toList)
    r.show()

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("fMeasureByLabel")

    println(eval.evaluate(r))

    spark.close()
    logger.info("Byes")
  }

  def loadData(spark: SparkSession): DataFrame = {
    val trainSet = Importer.importTrain()
    val data = trainSet
      .map(x => Row(x.categories.mkString(","), x.text))
//    val data = trainSet
//      .map(x => {
//        val h = MurmurHash3.stringHash(x.categories.mkString(",")) % 100
//        Row(h, x.text)
//      })

    val schema = StructType(Array(
      StructField("class", StringType, nullable=false),
      StructField("text", StringType, nullable=false)
    ))

    val rdd = spark.sparkContext.parallelize(data)
    spark.createDataFrame(rdd, schema)
  }

}
