# Problem 1

We need to first count the following values:

- True Positives (TP): Samples that are actually positive and were correctly classified as positive.

- True Negatives (TN): Samples that are actually negative and were correctly classified as negative.

- False Positives (FP): Samples that are actually negative but were incorrectly classified as positive.

- False Negatives (FN): Samples that are actually positive but were incorrectly classified as negative.

Counts: TP = 6, TN = 7, FP = 3, FN = 4.

Therefore:

- Precision = TP / (TP + FP) = 6 / (6 + 3) = 66.67%

- Recall = TP / (TP + FN) = 6 / (6 + 4) = 60.00%

- F-score = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.6667 * 0.6000) / (0.6667 + 0.6000) = 63.16%

- Accuracy = (TP + TN) / (TP + TN + FP + FN) = (6 + 7) / (6 + 7 + 3 + 4) = 65.00%

# Problem 2

Firstly, ensure the working directory is in the following structure

```
{working directory}/
    datasets_coursework1/
        real-state/
            test_full_Real-estate.csv
            train_full_Real-estate.csv
    part1_output/
        {empty}
    part1_train.py
    part1_test.py
```

Then, run the following command to train the models
```bash
python3 part1_train.py
```

After the training is complete, run the following command to test the models
```bash
python3 part1_test.py
```

The models will be saved under the `part1_output/` directory. The output metrics of the models on the test set will be:
```
Regression RMSE: 7.35
Classification Accuracy: 91.15%
```