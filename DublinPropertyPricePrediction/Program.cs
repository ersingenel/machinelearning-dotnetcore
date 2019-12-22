using DublinPropertyPricePrediction.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;

namespace DublinPropertyPricePrediction
{
    class Program
    {
        static string TrainingDataPath = @"Data/dublin-residential-property-price-index.csv";
        static MLContext mlContext = new MLContext(seed: 1);

        static void Main(string[] args)
        {
            // Create Model Using Training Data  
            var model = CreateModel();

            // Create Prediction Engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PropertyPrice, PropertyPricePrediction>(model);

            // Prepare Sample Data
            var sampleData = new List<PropertyPrice>{
                new PropertyPrice { Year = "August-2006", Price = 133.3f },
                new PropertyPrice { Year = "February-2012", Price = 54.5f },
                new PropertyPrice { Year = "October-2019", Price = 106.2f },
                new PropertyPrice { Year = "February-2020", Price = 0 }
            };

            // Make Prediction On Sample Data
            foreach (var sampleDataItem in sampleData)
            {
                var predictionResult = predictionEngine.Predict(sampleDataItem);
                var outputText = $"{sampleDataItem.Year} → Predicted Price: {predictionResult.Score}";
                if (sampleDataItem.Price > 0)
                {
                    outputText += $" Price: {sampleDataItem.Price}";
                }
                Console.WriteLine(outputText);
            }
            Console.ReadKey();
        }

        static ITransformer CreateModel()
        {
            // Load Data
            var trainingDataView = mlContext.Data.LoadFromTextFile<PropertyPrice>(
                                            path: TrainingDataPath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Data Process Configuration With Pipeline Data Transformations
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Year_tf", "Year")
                                      .Append(mlContext.Transforms.CopyColumns("Features", "Year_tf"))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

            // Set Training Algorithm 
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train Model
            var model = trainingPipeline.Fit(trainingDataView);

            return model;
        }
    }
}
