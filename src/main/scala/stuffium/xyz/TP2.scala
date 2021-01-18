package stuffium.xyz

import com.typesafe.scalalogging.LazyLogging

object TP2 extends LazyLogging {

  org.slf4j.LoggerFactory
    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
    .asInstanceOf[ch.qos.logback.classic.Logger]
    .setLevel(ch.qos.logback.classic.Level.INFO)

  def main(args: Array[String]): Unit = {
    logger.info("Hewos")

    logger.info("Byes")
  }

}
