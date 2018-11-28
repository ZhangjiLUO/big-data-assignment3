import org.apache.spark.sql.functions._
import org.apache.spark.sql.{
  SparkSession, SaveMode
}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{
  LDA, DistributedLDAModel
}
import org.apache.spark.ml.feature.{
  RegexTokenizer, StopWordsRemover, CountVectorizer, CountVectorizerModel
}

object TopicModeling {

  def main(args:Array[String]): Unit = {
    // environment set up
    val spark = SparkSession.builder().
        appName("TopicModeling").getOrCreate()
    import spark.implicits._

    // prompt user input and file paths
    if (args.length != 2) {
      println("Usage: <input_data_file> <output_folder>")
    }
    val inputDir = args(0)
    val outputDir = args(1)

    // convert sentiment to value
    val sentiment: String => Double = _ match {
      case "positive" => 5.0
      case "neutral" => 2.5
      case "negative" => 1.0
    }
    val converterOfSentiment = udf(sentiment)

    // remove special chars in text
    val specialStr = """[ ! @ # $ % ^ & * ( ) _ + - − , " ' ; : . "" ` ? &amp &amp; ]
        |@usairways @virginamerica @americanair @united @southwestair @delta""".
        stripMargin
    val specialChars = specialStr.split("\\s+")

    // extract input data for the columns
    val rawData = spark.read.option("header", "true").option("inferSchema", "true").
        csv(inputDir).select($"airline", $"text", $"airline_sentiment")
    
    val tweets = rawData.filter($"text" =!= "" && $"text".isNotNull).
        na.drop.withColumn("sentiment_values",
            converterOfSentiment($"airline_sentiment"))

    val avgSentiment = tweets.groupBy("airline").
        agg(avg("sentiment_values").alias("sentiment_average")).
        orderBy($"sentiment_average".desc)
    
    val bestAndWorst = tweets.filter($"airline" === "Virgin America" ||
        $"airline" === "US Airways")

    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words")
    val remover = new StopWordsRemover().
        setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")
    remover.setStopWords(remover.getStopWords ++ specialChars)
    val vectorizer = new CountVectorizer().setInputCol(remover.getOutputCol).
        setOutputCol("features").setVocabSize(2048)

    // set topics number to be 15
    val numTopics = 15

    // LDA set up
    val lda = new LDA().setK(numTopics).setMaxIter(50).setOptimizer("em")
    
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, vectorizer, lda))
    val pipelineModel = pipeline.fit(bestAndWorst)
    val vectorizerModel = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel]
    val ldaModel = pipelineModel.stages(3).asInstanceOf[DistributedLDAModel]

    // save the results to the output directory
    val vocabularys = vectorizerModel.vocabulary
    val termsWithId = udf {
      index:Seq[Int] => index.map(vocabularys(_))
    }
    val topics = ldaModel.describeTopics(maxTermsPerTopic=10).
        withColumn("terms", termsWithId(col("termIndices")))
    val stringify = udf((arr:Seq[String]) => arr.mkString(" "))
    val stringifyNum = udf((arr:Seq[Double]) => arr.mkString(" "))
    topics.withColumn("terms_stringify", stringify($"terms")).
        withColumn("termWeights_stringify", stringifyNum($"termWeights")).
        select($"topic", $"terms_stringify".alias("terms"),
            $"termWeights_stringify".alias("termWeights")).
        repartition(1).write.mode(SaveMode.Overwrite).
        option("header","true").format("csv").save(outputDir)
  }

}
