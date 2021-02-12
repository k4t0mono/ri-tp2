package xyz.stuffium

import com.typesafe.scalalogging.LazyLogging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification._
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

  Logger
    .getLogger("org")
    .setLevel(Level.ERROR)

  val indexer: StringIndexer = new StringIndexer()
    .setInputCol("class")
    .setOutputCol("label")

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

    val (train, test) = getData(spark, fit=false)

    val cls = Classifier.DT
    val model = trainClassifier(train, cls, fit=true)
    val e = testModel(model, test, cls)
    println(e)

    spark.close()
    logger.info("Byes")
  }

  def testModel(model: CrossValidatorModel, test: DataFrame, classifier: Classifier): Double = {
    val prediction = model.transform(test)

    prediction
      .select("class","label", "prediction")
      .coalesce(1)
      .write
      .option("header", true)
      .csv(s"results/$classifier")

    evaluator.evaluate(prediction)
  }

  def getData(spark: SparkSession, fit: Boolean): (DataFrame, DataFrame) = {
    val train_raw = loadData(spark)
    val test_raw = loadData(spark, test=true)

    val preprocess = if (fit) {
      val data_raw = train_raw.union(test_raw)

      val pre_pipe = new Pipeline()
        .setStages(Array(indexer, tokenizer, remover, hashingTF, idf))

      val m = pre_pipe.fit(data_raw)
      m.write.overwrite().save("./cache/PP")

      m
    } else {
      PipelineModel.load("./cache/PP")
    }

    preprocess.write.overwrite.save("./cache/preprocess")

    val trainDF = preprocess.transform(train_raw)
    val testDF = preprocess.transform(test_raw)

    (trainDF, testDF)
  }

  def trainClassifier(train: DataFrame, classifier: Classifier, fit: Boolean): CrossValidatorModel = {
    if (fit) {
      val m = classifier match {
        case Classifier.NB => trainNB(train)
        case Classifier.DT => trainDT(train)
        case Classifier.RF => trainRF(train)
      }

      m.write.overwrite().save(s"cache/$classifier")
      m
    } else {
      CrossValidatorModel.load(s"cache/$classifier")
    }
  }

  def trainRF(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Decision Tree ")

    val rf = new RandomForestClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, rf))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.percentile, Array(0.1, 0.2, 0.15))
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

  def trainDT(dataFrame: DataFrame): CrossValidatorModel = {
    logger.info("Testando Decision Tree ")

    val dt = new DecisionTreeClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(chiSq, scaler, dt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(chiSq.percentile, Array(0.1, 0.2, 0.15))
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

  def trainNB(dataFrame: DataFrame): CrossValidatorModel = {
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
