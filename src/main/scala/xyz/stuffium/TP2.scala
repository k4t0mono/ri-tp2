package xyz.stuffium

import com.typesafe.scalalogging.LazyLogging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import xyz.stuffium.TP2.Classifier.Classifier
import xyz.stuffium.util.Importer

object TP2 extends LazyLogging {

  org.slf4j.LoggerFactory
    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
    .asInstanceOf[ch.qos.logback.classic.Logger]
    .setLevel(ch.qos.logback.classic.Level.INFO)

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("tokensSW")

  val remover: StopWordsRemover = new StopWordsRemover()
    .setLocale("en_US")
    .setInputCol("tokensSW")
    .setOutputCol("tokens")

  val hashingTF: HashingTF = new HashingTF()
    .setInputCol("tokens")
    .setOutputCol("rawTF")

  val idf: IDF = new IDF()
    .setInputCol("rawTF")
    .setOutputCol("tf-idf")
    .setMinDocFreq(2)

  val chiSq: ChiSqSelector = new ChiSqSelector()
    .setFeaturesCol("tf-idf")
    .setLabelCol("label")
    .setOutputCol("features")

  val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

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
      .fit(trainRaw)

    val train = indexer.transform(trainRaw)
//    val test = indexer.transform(loadData(spark, test=true))

//    Classifier.tags()
      Seq(Classifier.DT)
      .foreach(x => {
        val m = test(train, x).get
        val p = m.bestModel.transform(train)
        val e = evaluator.evaluate(p)
        logger.error(s"$e")
        println(e)
    })

//    model.write.overwrite.save("./models/nb_cv")
//    val model = CrossValidatorModel.load("./models/nb_cv")
//
//    val r = model.transform(train)
//    println(model.bestModel.params.toList)
//    r.show()
//
//    val eval = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("fMeasureByLabel")
//
//    val rt = model.transform(train)
//    rt.cache()
//
//    rt
//      .select("label", "prediction")
//      .write
//      .mode(SaveMode.Overwrite)
//      .csv("./aaa2.csv")

    spark.close()
    logger.info("Byes")
  }

  def test(dataFrame: DataFrame, classifier: Classifier): Option[CrossValidatorModel] = {
    classifier match {
      case Classifier.NB => Some(test_nb(dataFrame))
      case Classifier.SVM => Some(test_svm(dataFrame))
      case Classifier.DT => Some(test_dt(dataFrame))
      case _ => None
    }
  }

  def test_svm(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Naive Bayes")

    val svm = new LinearSVC()
    val dt = new DecisionTreeClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, chiSq, dt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100, 200, 500, 1000))
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200, 500))
      .addGrid(svm.maxIter, Array(10, 100, 200))
      .addGrid(svm.regParam, Array(0.1, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    logger.info("Fim do treinamento")

    cv.fit(dataFrame)
  }

  def test_nb(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Naive Bayes")
    val nb = new NaiveBayes()

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, chiSq, nb))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100, 200, 500, 1000))
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200, 500))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    logger.info("Fim do treinamento")

    cv.fit(dataFrame)
  }

  def test_dt(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Decision Tree")

    val dt = new DecisionTreeClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, chiSq, dt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100, 200, 500, 1000))
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200, 500))
      .addGrid(dt.maxBins, Array(16, 32, 64, 128))
      .addGrid(dt.maxDepth, Array(5, 10, 15, 20))
      .addGrid(dt.minInfoGain, Array(0.0, 0.1, 0.2, 0.05, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    logger.info("Fim do treinamento")

    cv.fit(dataFrame)
  }

  def loadData(spark: SparkSession, test: Boolean = false): DataFrame = {
    logger.info(s"loading data $test")
    val rawData = Importer.importData(test)
    val data = rawData
      .map(x => Row(x.categories.mkString(","), x.text))

    val schema = StructType(Array(
      StructField("class", StringType, nullable=false),
      StructField("text", StringType, nullable=false)
    ))

    val rdd = spark.sparkContext.parallelize(data)
    spark.createDataFrame(rdd, schema)
  }

  object Classifier extends Enumeration {
    type Classifier = Value
    val NB, DT, SVM, MLP = Value

    def tags(): List[Classifier] = {
      List(NB, DT, SVM, MLP)
    }
  }

}
