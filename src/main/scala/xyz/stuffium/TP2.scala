package xyz.stuffium

import org.apache.log4j.{Level, Logger}
import com.typesafe.scalalogging.LazyLogging
import xyz.stuffium.util.Importer

object TP2 extends LazyLogging {

  org.slf4j.LoggerFactory
    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
    .asInstanceOf[ch.qos.logback.classic.Logger]
    .setLevel(ch.qos.logback.classic.Level.INFO)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("xyz").setLevel(Level.WARN)
    logger.info("Hewos")

//    val sc = new SparkContext(master="local[*]", appName = "RITP2")

    val trainSet = Importer.importTrain()
    println(trainSet)

    logger.info("Byes")
  }

}
