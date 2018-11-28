import org.apache.spark.sql.{
  SparkSession, SaveMode
}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.fpm.FPGrowth

object FrequentItem {

  def main(args:Array[String]): Unit = {
    // environment set up
    val spark = SparkSession.builder().
      appName("FrequentItem").getOrCreate()
    import spark.implicits._

    // prompt user input and file paths
    if (args.length != 2) {
      println("Usage: <input_data_file> <output_folder>")
    }
    val inputDir = args(0)
    val outputDir = args(1)

    // get transactions
    val transactions = spark.read.option("header", "true").
      option("inferSchema", "true").csv(inputDir).
      select($"order_id", $"product_id")

    // process data
    val vectorized = transactions.groupBy("order_id").
      agg(collect_list("product_id") as "items").
      orderBy($"order_id").select($"items")

    // build fpgGrowth model
    val fpGrowth = new FPGrowth().setItemsCol("items").
      setMinSupport(0.01).setMinConfidence(0.01)
    val model = fpGrowth.fit(vectorized)
    val stringify = udf((arr: Seq[Int]) => arr.mkString(" "))

    model.freqItemsets.withColumn("items_stringify",
      stringify($"items")).select($"items_stringify".
        alias("items"), $"freq").limit(10).repartition(1).
      write.mode(SaveMode.Overwrite).option("header","true").
      format("csv").save(outputDir + "/freqItemSets")

    // show associated rules
    model.associationRules.orderBy($"confidence".desc).
      withColumn("antecedent_stringify", stringify($"antecedent")).
      withColumn("consequent_stringify", stringify($"consequent")).
      select($"antecedent_stringify".alias("antecedent"),
        $"consequent_stringify".alias("consequent"), $"confidence").
      limit(10).repartition(1).write.mode(SaveMode.Append).
      option("header", "true").format("csv").save(outputDir + "/associationRules")
  }

}
