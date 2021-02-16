package org.apache.spark.ml.tuning

import org.apache.spark.ml.Model
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.ThreadUtils

import scala.concurrent.Future
import scala.concurrent.duration.Duration

class CrossValidatorKME(override val uid: String) extends CrossValidator {

  var splitsCV: Array[(RDD[Row], RDD[Row])] = _

  def this() = this(Identifiable.randomUID("cvKME"))

  override def fit(dataset: Dataset[_]): CrossValidatorModel = instrumented { instr =>
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)

    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContext

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, numFolds, seed, parallelism, foldCol)
    logTuningParams(instr)

    val collectSubModelsParam = $(collectSubModels)

    val subModels: Option[Array[Array[Model[_]]]] = if (collectSubModelsParam) {
      Some(Array.fill($(numFolds))(Array.ofDim[Model[_]](epm.length)))
    } else None

    // Compute metrics for each model over each split
    val (splits, schemaWithoutFold) = if ($(foldCol) == "") {
      (MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed)), schema)
    } else {
      val filteredSchema = StructType(schema.filter(_.name != $(foldCol)).toArray)
      (MLUtils.kFold(dataset.toDF, $(numFolds), $(foldCol)), filteredSchema)
    }

    splitsCV = splits

    val metrics = splits.zipWithIndex.map { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schemaWithoutFold).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schemaWithoutFold).cache()
      instr.logInfo(s"Train split $splitIndex with multiple sets of parameters.")

      // Fit models in a Future for training in parallel
      val foldMetricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        Future[Double] {
          val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]
          if (collectSubModelsParam) {
            subModels.get(splitIndex)(paramIndex) = model
          }
          // TODO: duplicate evaluator to take extra params from input
          val metric = eval.evaluate(model.transform(validationDataset, paramMap))
          instr.logInfo(s"Got metric $metric for model trained with $paramMap.")
          metric
        } (executionContext)
      }

      // Wait for metrics to be calculated
      val foldMetrics = foldMetricFutures.map(ThreadUtils.awaitResult(_, Duration.Inf))

      // Unpersist training & validation set once all metrics have been produced
      trainingDataset.unpersist()
      validationDataset.unpersist()
      foldMetrics
    }.transpose.map(_.sum / $(numFolds)) // Calculate average metric over all splits

    instr.logInfo(s"Average cross-validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    instr.logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    instr.logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new CrossValidatorModel(uid, bestModel, metrics)
      .setSubModels(subModels).setParent(this))
  }

}
