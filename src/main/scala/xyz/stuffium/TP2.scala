package xyz.stuffium

import com.typesafe.scalalogging.LazyLogging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
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

    val trainSet = Importer.importTrain()
    val data = trainSet
      .map(x => Row(x.categories.mkString(","), x.text))

    val schema = StructType(Array(
      StructField("class", StringType, nullable=false),
      StructField("text", StringType, nullable=false)
    ))

    val rdd = spark.sparkContext.parallelize(data)
    val df = spark.createDataFrame(rdd, schema)

    val indexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")

    val hashingTF = new HashingTF()
      .setInputCol("tokens")
      .setOutputCol("rawTF")
      .setNumFeatures(100)

    val idf = new IDF()
      .setInputCol("rawTF")
      .setOutputCol("tf-idf")
      .setMinDocFreq(2)

    val chiSq = new ChiSqSelector()
      .setFeaturesCol("tf-idf")
      .setLabelCol("label")
      .setOutputCol("features")

    val naiveBayes = new NaiveBayes()

    val pipeline = new Pipeline()
      .setStages(Array(indexer, tokenizer, hashingTF, idf, chiSq, naiveBayes))

    val model = pipeline.fit(df)
    model.write.overwrite().save("/tmp/spark-logistic-regression-model")

    val prediction = model
      .transform(df)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(prediction)
    println(accuracy)

    //    val r = model.transform(df).select("tf-idf", "features").collect().last
//    println(r)

    spark.close()
    logger.info("Byes")
  }

}
