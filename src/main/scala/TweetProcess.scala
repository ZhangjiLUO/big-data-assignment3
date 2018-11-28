import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{
  ParamGridBuilder, CrossValidator
}
import org.apache.spark.ml.feature.{
  HashingTF, StopWordsRemover, StringIndexer, Tokenizer
}
import org.apache.spark.ml.classification.{
  NaiveBayes, RandomForestClassifier
}
import org.apache.spark.sql.{
  SparkSession, SaveMode
}

object TweetProcess {

  def main(args:Array[String]): Unit = {

    val spark = SparkSession.builder().
        appName("TweetProcessing").getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    
    if (args.length != 2) {
      println("Usage: inputDir outputDir")
    }
    val inputDir = args(0). // input path
    val outputDir = args(1). // output path

    val df = spark.read.option("header", "true").option("inferSchema", "true").
        csv(inputDir).select($"text", $"airline_sentiment")

    val tweet = df.filter($"text" =!= "" && $"text".isNotNull)

    // split data
    val Array(training, test) = tweet.randomSplit(Array(0.8, 0.2), seed = 1111)

    // tokenizer
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    // stopwordsRemover
    val remover = new StopWordsRemover().
        setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")

    // hashingTF
    val hashTF = new HashingTF().
        setInputCol(remover.getOutputCol).setOutputCol("features")

    // StringIndexer
    val indexer = new StringIndexer().
        setInputCol("airline_sentiment").setOutputCol("label")

    // two models: random forests (rf) and naive bayes (nb)
    val rf = new RandomForestClassifier().
        setLabelCol("label").setFeaturesCol("features")
    val nb = new NaiveBayes().setModelType("multinomial").
        setSmoothing(1.0).setLabelCol("label").setFeaturesCol("features")

    // pipeline
    val pipelineRF = new Pipeline().
        setStages(Array(tokenizer, remover, hashTF, indexer, rf))
    val pipelineNB = new Pipeline().
        setStages(Array(tokenizer, remover, hashTF, indexer, nb))

    // parameterGridBuilder (pgb) for parameter tuning
    val pgbRF = new ParamGridBuilder().
        addGrid(hashTF.numFeatures, Array(100, 1000)).
        addGrid(rf.numTrees, Array(10,200)).build()
    val pgbNB = new ParamGridBuilder().
        addGrid(hashTF.numFeatures, Array(100, 1000)).build()

    // set up CrossValidator to compare the performance of models
    val evaluator = new MulticlassClassificationEvaluator().
        setLabelCol("label").setPredictionCol("prediction").
        setMetricName("accuracy")
    val cvRF = new CrossValidator().setEstimator(pipelineRF).
        setEvaluator(evaluator).setEstimatorParamMaps(pgbRF).setNumFolds(10)
    val cvNB = new CrossValidator().setEstimator(pipelineNB).
        setEvaluator(evaluator).setEstimatorParamMaps(pgbNB).setNumFolds(10)

    // train model
    val cvRF_model = cvRF.fit(training)
    val cvNB_model = cvNB.fit(training)
    // predict
    val predictRF = cvRF_model.transform(test)
    val predictNB = cvNB_model.transform(test)

    // find the accuracy
    val accuracyRF = evaluator.evaluate(predictRF)
    val accuracyNB = evaluator.evaluate(predictNB)
    val evaluationDF = sc.parallelize(Seq(("Random Forest", accuracyRF),
        ("Naive Bayes", accuracyNB))).toDF("model", "accuracy")

    // save result
    evaluationDF.repartition(1).write.
        mode(SaveMode.Overwrite).option("header","true").
        format("csv").save(outputDir)
  }

}
