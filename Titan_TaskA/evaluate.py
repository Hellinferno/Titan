"""
Evaluation Script for Titan Track A
Usage: python evaluate.py --pred results.csv --gold data/train.csv
"""
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate(pred_path, gold_path):
    print(f"DTO Eval: {pred_path} vs {gold_path}")
    
    try:
        pred_df = pd.read_csv(pred_path)
        gold_df = pd.read_csv(gold_path)
        
        # Normalize columns
        if 'id' in gold_df.columns: gold_df.rename(columns={'id': 'StoryID'}, inplace=True)
        if 'label' in gold_df.columns: gold_df.rename(columns={'label': 'Prediction'}, inplace=True)
        
        # Merge on StoryID
        merged = pd.merge(pred_df, gold_df, on='StoryID', suffixes=('_pred', '_gold'))
        
        if len(merged) == 0:
            print("❌ No overlapping StoryIDs found!")
            return
            
        y_pred = merged['Prediction_pred'].astype(int)
        y_true = merged['Prediction_gold'].astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n📊 Evaluation Results:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print("\n   Confusion Matrix:")
        print(f"   TN: {cm[0][0]}  FP: {cm[0][1]}")
        print(f"   FN: {cm[1][0]}  TP: {cm[1][1]}")
        
    except Exception as e:
        print(f"❌ Evaluation Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()
    evaluate(args.pred, args.gold)
