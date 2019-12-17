using Microsoft.ML.Data;

namespace DublinPropertyPricePrediction.Models
{
    public class PropertyPrice
    {
        [ColumnName("Year"), LoadColumn(0)]
        public string Year { get; set; }


        [ColumnName("Price"), LoadColumn(1)]
        public float Price { get; set; }
    }
}