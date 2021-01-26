package xyz.stuffium.util

case class ReutersArticle(id: Int, text: String) {
  var categories = List("")

  override def toString: String = s"<RA id=$id cat=$categories />"

  override def equals(obj: Any): Boolean = super.equals(obj)
  
  def update(cats: List[String]): Unit = {
    categories = cats
  }
  
}
