package xyz.stuffium

import com.typesafe.scalalogging.LazyLogging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import xyz.stuffium.TP2.Classifier.Classifier
import xyz.stuffium.util.Importer

object TP2 extends LazyLogging {

  org.slf4j.LoggerFactory
    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
    .asInstanceOf[ch.qos.logback.classic.Logger]
    .setLevel(ch.qos.logback.classic.Level.INFO)

  Logger
    .getLogger("org")
    .setLevel(Level.ERROR)

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
    .setOutputCol("featuresRaw")

  val scaler: MinMaxScaler = new MinMaxScaler()
    .setInputCol("featuresRaw")
    .setOutputCol("features")

  val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  def main(args: Array[String]): Unit = {
    logger.info("Hewos")

    val spark = SparkSession
      .builder()
      .appName("ri-tp2")
      .master("local[*]")
      .getOrCreate()

    val indexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")

//    val preprocess = new Pipeline()
//      .setStages(Array(tokenizer, remover, indexer, hashingTF, idf))
//    val model = preprocess.fit(trainRaw)

    val preprocess = PipelineModel.load("./cache/preprocess")
    preprocess.write.overwrite.save("./cache/preprocess")

    val train = preprocess.transform(loadData(spark))
    train.cache()

//    val test = model.transform(loadData(spark, test=true))
//    test.cache()

      Seq(Classifier.NB)
      .foreach(x => {
        logger.info(s"Trying $x")

        val m = testClassifier(train, x).get
        m.write.overwrite().save(s"models/${x}_n2")
        logger.error(s"Trying $x")
        println(m.bestModel.explainParams())

//        val m = CrossValidatorModel.load(s"models/${x}_n2")
//        println(m.bestModel)

        val pred = m.transform(train)
        val eval = evaluator.evaluate(pred)

//        pred
//          .select("label", "prediction")
//          .coalesce(1)
//          .write
//          .csv(s"./results/$x.csv")

        logger.error(s"$eval")
        println(eval)
    })

    spark.close()
    logger.info("Byes")
  }

  def testClassifier(dataFrame: DataFrame, classifier: Classifier): Option[CrossValidatorModel] = {
    classifier match {
      case Classifier.NB => Some(test_nb(dataFrame))
      case Classifier.DT => Some(test_dt(dataFrame))
      case Classifier.RF => Some(test_rf(dataFrame))
      case _ => None
    }
  }

  def test_rf(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Decision Tree ")

    val rf = new RandomForestClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, rf))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200))
      .addGrid(rf.maxDepth, Array(5, 10, 15))
      .addGrid(rf.numTrees, Array(5, 10, 15))
//      .addGrid(rf.maxBins, Array(32, 16, 64))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(2)

    logger.info("Fim do treinamento")

    cv.fit(dataFrame)
  }

  def test_dt(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Decision Tree ")

    val dt = new DecisionTreeClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, dt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200))
      .addGrid(dt.maxDepth, Array(5, 10, 15))
      .addGrid(dt.maxBins, Array(32, 16, 64))
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
      .setStages(Array(chiSq, scaler, nb))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.percentile, Array(0.1, 0.2, 0.15))
      .addGrid(nb.modelType, Array("multinomial"))
      .addGrid(nb.smoothing, Array(1.0, 0.5, 1.5))
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
    val NB, DT, RF = Value

    def tags(): List[Classifier] = {
      List(NB, DT, RF)
    }
  }

}
