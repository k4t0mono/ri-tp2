package xyz.stuffium

import com.typesafe.scalalogging.LazyLogging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import xyz.stuffium.TP2.Classifier.{Classifier, DT}
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
    .setOutputCol("featuresRaw")

  val scaler: MinMaxScaler = new MinMaxScaler()
    .setInputCol("featuresRaw")
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

    val preprocess = new Pipeline()
      .setStages(Array(tokenizer, remover, indexer, hashingTF, idf))

//    val model = preprocess.fit(trainRaw)
    val model = PipelineModel.load("./cache/preprocess")
    model.write.overwrite.save("./cache/preprocess")
    val train = model.transform(trainRaw)
    train.cache()

//    Classifier.tags()
      Seq(Classifier.DT)
      .foreach(x => {
//        val m = test(train, x).get
//        m.write.overwrite().save(s"models/$x")
        val m = CrossValidatorModel.load(s"models/$x")
        println(m.explainParams())
        println(m.estimator.toString())
        val p = m.transform(train)
//        val p = validateBestModel(train, train, x, m).get
        val e = evaluator.evaluate(p)

        p
          .select("label", "prediction")
          .coalesce(1)
          .write
          .csv(s"./results/$x.csv")

        logger.error(s"$e")
        println(e)
    })

    spark.close()
    logger.info("Byes")
  }

  def validateBestModel(train: DataFrame, test: DataFrame, classifier: Classifier, model: CrossValidatorModel): Option[DataFrame] = {
    classifier match {
      case DT =>
        println(model.bestModel.extractParamMap())
//        val dt = new DecisionTreeClassifier()
//        val pipeline = new Pipeline()
//          .setStages(Array(chiSq, scaler, dt))
//        val m = pipeline.fit(train)
//        Some(m.transform(test))
        None
      case _ => None
    }
  }

  def test(dataFrame: DataFrame, classifier: Classifier): Option[CrossValidatorModel] = {
    classifier match {
      case Classifier.NB => Some(test_nb(dataFrame))
      case Classifier.SVM => Some(test_svm(dataFrame))
      case Classifier.LR => Some(test_lr(dataFrame))
      case Classifier.DT => Some(test_dt(dataFrame))
      case _ => None
    }
  }

  def test_dt(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Decision Tree ")

    val dt = new DecisionTreeClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, dt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200, 500))
      .addGrid(dt.maxDepth, Array(5, 10, 15))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    logger.info("Fim do treinamento")

    cv.fit(dataFrame)
  }

  def test_lr(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Logistic ")

    val lr = new LogisticRegression()

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.numTopFeatures, Array(50, 100, 200, 500))
      .addGrid(lr.maxIter, Array(10, 100, 200))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    logger.info("Fim do treinamento")

    cv.fit(dataFrame)
  }

  def test_svm(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Naive Bayes")

    val svm = new LinearSVC()
    val ova = new OneVsRest()
      .setClassifier(svm)

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, ova))

    val paramGrid = new ParamGridBuilder()
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
      .setStages(Array(chiSq, scaler, nb))

    val paramGrid = new ParamGridBuilder()
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
    val NB, LR, SVM, MLP, DT, RF = Value

    def tags(): List[Classifier] = {
      List(NB, LR, SVM, MLP, DT, RF)
    }
  }

}
