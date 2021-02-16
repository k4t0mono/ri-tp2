package xyz.stuffium

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MLReadable, MLReader}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class Cacher(val uid: String) extends Transformer with DefaultParamsWritable {
  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(s"Caching data")
    dataset.select("featuresRaw", "label").toDF.cache()
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  def this() = this(Identifiable.randomUID("CacherTransformer"))
}

object Cacher extends MLReadable[Cacher] {
  override def read: MLReader[Cacher] = new CacherReader

  private class CacherReader extends MLReader[Cacher] {
    override def load(path: String): Cacher = new Cacher
  }
}

