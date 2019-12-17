using DublinPropertyPricePrediction.Models;
using Microsoft.ML;
using System;

namespace DublinPropertyPricePrediction
{
    class Program
    {
        static string TrainDataPath = @"Data/dublin-residential-property-price-index.csv";
        static MLContext mlContext = new MLContext(seed: 1);

        static void Main(string[] args)
        {
            // Load Data
            var trainingDataView = mlContext.Data.LoadFromTextFile<PropertyPrice>(
                                            path: TrainDataPath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Year_tf", "Year")
                                      .Append(mlContext.Transforms.CopyColumns("Features", "Year_tf"))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm 
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train Model
            var model = trainingPipeline.Fit(trainingDataView);

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PropertyPrice, PropertyPricePrediction>(model);

            var sampleData = new PropertyPrice { Year = "October-2020", Price = 0 };
            
            // Use model to make prediction on input data
            var predictionResult = predictionEngine.Predict(sampleData);

            Console.WriteLine($"Year: {sampleData.Year} Predicted Price: {predictionResult.Score}");
            Console.ReadKey();
        }
    }
}
