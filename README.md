[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmaxerh%2FTSAD_evaluation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Time Series Anomaly Detection - Evaluation
1. combine multiple csv files into one. CSV must contain columns:
    - algorithm: name of the algorithm
    - dataset: name of the dataset
    - entity: name of the entity in the dataset
    - seq_len: input sequence length
    - TP: true positives
    - TN: true negatives
    - FP: false positives
    - FN: false negatives
2. calculate P,R,F1,ROC/AUC for datasets and algorithms
3. output tables in command line

## How to start
```bash
python main.py -a Example_Example2 -d myds
```