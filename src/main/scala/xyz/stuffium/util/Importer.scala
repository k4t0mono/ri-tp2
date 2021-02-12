package xyz.stuffium.util

import com.typesafe.scalalogging.LazyLogging

import java.io.File
import java.nio.charset.MalformedInputException
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

object Importer extends LazyLogging {

  val articlesDict: mutable.HashMap[Int, ReutersArticle] = new mutable.HashMap[Int, ReutersArticle]
  val articles: ListBuffer[ReutersArticle] = new ListBuffer[ReutersArticle]

  def importData(test: Boolean, legacy: Boolean = false): List[ReutersArticle] = {
    logger.debug("Importing train data")

    val trainDir = if (test) {
      new File("data/test")
    } else {
      new File("data/training")
    }

    trainDir
      .listFiles()
      .toList
      .foreach(x => loadFilesCat(x))

    if(legacy) {
      articlesDict
        .iterator
        .map(x => {
          x._2.update(ArticleCategories.get(x._1))
          x._2
        })
        .toList
        .sortBy(x => x.id)
    } else {
      articles
        .toList
    }
  }

  def loadFilesCat(path: File): Unit = {
    logger.trace(s"Visiting ${path.toString}")
    val cat = path.toString.split("/").last

    path
      .listFiles()
      .toList
      .foreach(x => {
        try {
          importFile(x, cat)
        } catch {
          case e: MalformedInputException => logger.error(s"Could not read ${path.toString}, ${e.toString}")
        }
      })
  }

  def importFile(fl: File, cat: String): Unit = {
    val id = fl.toString.split("/").last.toInt
    val text = readFile(fl.toString)

    val ac = ReutersArticle(id, text)
    ac.update(List(cat))

    articles += ac
  }

  def importFileOld(fl: File, cat: String): Unit = {
    val id = fl.toString.split("/").last.toInt
//    val text = PreProcessor.treatData(readFile(fl.toString))
    val text = readFile(fl.toString)

    ArticleCategories.add(id, cat)
    articlesDict.put(id, ReutersArticle(id, text))
  }

  def readFile(path: String): String = {
    logger.trace(s"readFile($path)")
    val buff = Source.fromFile(path)
    val lines = buff.getLines().toList
    buff.close()

    val text = lines
      .map(x => x.trim)
      .filter(x => x.nonEmpty)
      .mkString(" ")

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
