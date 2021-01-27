name := "ri-tp2"

version := "0.1"

scalaVersion := "2.12.13"

libraryDependencies ++= Seq(
  // Logging
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2",

  // Machine learning
  "org.apache.spark" %% "spark-core" % "3.1.0",
  "org.apache.spark" %% "spark-mllib" % "3.1.0",
  "org.apache.spark" %% "spark-sql" % "3.1.0",

// NLP
  "org.apache.opennlp" % "opennlp-tools" % "1.9.3",
)
