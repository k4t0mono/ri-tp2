package xyz.stuffium.util

import com.typesafe.scalalogging.LazyLogging

import java.io.File
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

object Importer extends LazyLogging {

  val articlesDict: mutable.HashMap[Int, ReutersArticle] = new mutable.HashMap[Int, ReutersArticle]()

  def importData(test: Boolean): List[ReutersArticle] = {
    logger.debug("Importing train data")

    val trainDir = test match {
      case false => new File("data/training")
      case true => new File("data/test")
    }

    trainDir
      .listFiles()
      .toList
      .foreach(x => loadFilesCat(x))

    articlesDict
      .iterator
      .map(x => {
        x._2.update(ArticleCategories.get(x._1))
        x._2
      })
      .toList
      .sortBy(x => x.id)
  }

  def loadFilesCat(path: File): Unit = {
    logger.trace(s"Visiting ${path.toString}")
    val cat = path.toString.split("/").last

    path
      .listFiles()
      .toList
      .foreach(x => importFile(x, cat))
  }

  def importFile(fl: File, cat: String): Unit = {
    val id = fl.toString.split("/").last.toInt
//    val text = PreProcessor.treatData(readFile(fl.toString))
    val text = readFile(fl.toString)

    ArticleCategories.add(id, cat)
    articlesDict.put(id, ReutersArticle(id, text))
  }

  def readFile(path: String): String = {
    logger.trace(s"readFile($path)")
    val buff = Source.fromFile(path)

    val text = buff
      .getLines()
      .map(x => x.trim)
      .mkString(" ")

    buff.close()

    text
  }

  object ArticleCategories {
    val dict: mutable.HashMap[Int, ListBuffer[String]] = new mutable.HashMap[Int, ListBuffer[String]]()

    def add(id: Int, cat: String): Unit = {
      val l = dict.get(id) match {
        case None => new ListBuffer[String]
        case Some(l) => l
      }

      l += cat
      dict.put(id, l)
    }

    def get(id: Int): List[String] = {
      dict.get(id) match {
        case Some(l) => l.toList
        case None => List()
      }
    }
  }

}
