import unittest
import pandas as pd
import os
import csv
from classifier import get_model_name

class TestTitanSubmission(unittest.TestCase):
    
    def test_results_file_exists(self):
        """Check results.csv exists"""
        self.assertTrue(os.path.exists("results.csv"), "results.csv missing")
        
    def test_results_schema(self):
        """Check output columns match strict requirement"""
        if os.path.exists("results.csv"):
            df = pd.read_csv("results.csv")
            expected = {"StoryID", "Prediction", "Rationale"}
            self.assertEqual(set(df.columns), expected, "Bad Schema")
            
    def test_predictions_binary(self):
        """Check all predictions are 0 or 1"""
        if os.path.exists("results.csv"):
            df = pd.read_csv("results.csv")
            valid_preds = {0, 1}
            # Check values are in set {0, 1}
            for p in df['Prediction']:
                self.assertIn(p, valid_preds, f"Invalid prediction value: {p}")

    def test_model_selection(self):
        """Check model selection logic"""
        os.environ["USE_OLLAMA"] = "true"
        os.environ["OLLAMA_MODEL"] = "gemma2:2b"
        self.assertEqual(get_model_name(), "gemma2:2b")
        
        os.environ["USE_OLLAMA"] = "false"
        self.assertEqual(get_model_name(), "anthropic/claude-3.5-sonnet")


if __name__ == '__main__':
    unittest.main()
