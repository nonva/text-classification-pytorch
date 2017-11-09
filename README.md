# text-classification-pytorch
cnn and rnn for chinese text classification using pytorch

## Movie Review dataset

### CNN

This is roughly the same model as [CNN for text classification](https://arxiv.org/pdf/1408.5882.pdf).

Run `python mr_cnn.py` for multiple times, the average accuracy is about 75%, very close to the original paper.

Result:

```
Loading data...
Training: 9595, Testing: 1067, Vocabulary: 8000
Configuring CNN model...
TextCNN (
  (embedding): Embedding(8000, 128)
  (convs): ModuleList (
    (0): Conv1d(128, 100, kernel_size=(3,), stride=(1,))
    (1): Conv1d(128, 100, kernel_size=(4,), stride=(1,))
    (2): Conv1d(128, 100, kernel_size=(5,), stride=(1,))
  )
  (fc1): Linear (300 -> 2)
  (dropout): Dropout (p = 0.5)
)
Epoch: 1
Iter:    100, Train Loss:   0.74, Train Acc:  60.00%, Val Loss:   0.65, Val Acc:  62.32%, Time: 0:00:06 *
Epoch: 2
Iter:    200, Train Loss:   0.68, Train Acc:  76.00%, Val Loss:   0.63, Val Acc:  65.98%, Time: 0:00:08 *
Iter:    300, Train Loss:   0.59, Train Acc:  66.00%, Val Loss:   0.62, Val Acc:  66.07%, Time: 0:00:11 *
Epoch: 3
Iter:    400, Train Loss:   0.58, Train Acc:  86.00%, Val Loss:   0.59, Val Acc:  67.48%, Time: 0:00:13 *
Iter:    500, Train Loss:   0.51, Train Acc:  82.00%, Val Loss:   0.59, Val Acc:  68.42%, Time: 0:00:16 *
Epoch: 4
Iter:    600, Train Loss:   0.48, Train Acc:  74.00%, Val Loss:   0.56, Val Acc:  70.76%, Time: 0:00:19 *
Iter:    700, Train Loss:   0.42, Train Acc:  78.00%, Val Loss:   0.57, Val Acc:  70.67%, Time: 0:00:21
Epoch: 5
Iter:    800, Train Loss:    0.4, Train Acc:  76.00%, Val Loss:   0.58, Val Acc:  70.20%, Time: 0:00:24
Iter:    900, Train Loss:   0.34, Train Acc:  84.00%, Val Loss:    0.6, Val Acc:  72.35%, Time: 0:00:27 *
Epoch: 6
Iter:   1000, Train Loss:   0.31, Train Acc:  82.00%, Val Loss:   0.62, Val Acc:  71.98%, Time: 0:00:29
Iter:   1100, Train Loss:   0.26, Train Acc: 100.00%, Val Loss:   0.61, Val Acc:  72.91%, Time: 0:00:32 *
Epoch: 7
Iter:   1200, Train Loss:   0.24, Train Acc:  94.00%, Val Loss:    0.6, Val Acc:  74.79%, Time: 0:00:34 *
Iter:   1300, Train Loss:   0.22, Train Acc:  96.00%, Val Loss:   0.71, Val Acc:  71.04%, Time: 0:00:37
Epoch: 8
Iter:   1400, Train Loss:   0.18, Train Acc:  98.00%, Val Loss:   0.65, Val Acc:  74.04%, Time: 0:00:40
Iter:   1500, Train Loss:   0.15, Train Acc:  96.00%, Val Loss:   0.68, Val Acc:  74.32%, Time: 0:00:42
Epoch: 9
Iter:   1600, Train Loss:   0.17, Train Acc:  94.00%, Val Loss:   0.69, Val Acc:  73.76%, Time: 0:00:45
Iter:   1700, Train Loss:   0.13, Train Acc:  98.00%, Val Loss:   0.72, Val Acc:  74.23%, Time: 0:00:48
Epoch: 10
Iter:   1800, Train Loss:   0.11, Train Acc: 100.00%, Val Loss:   0.76, Val Acc:  74.51%, Time: 0:00:50
Iter:   1900, Train Loss:    0.1, Train Acc:  94.00%, Val Loss:   0.78, Val Acc:  74.51%, Time: 0:00:53
Epoch: 11
Iter:   2000, Train Loss:  0.089, Train Acc: 100.00%, Val Loss:   0.81, Val Acc:  74.51%, Time: 0:00:55
Iter:   2100, Train Loss:  0.084, Train Acc:  96.00%, Val Loss:   0.85, Val Acc:  74.04%, Time: 0:00:58
Epoch: 12
Iter:   2200, Train Loss:  0.074, Train Acc:  98.00%, Val Loss:   0.87, Val Acc:  74.23%, Time: 0:01:01
Iter:   2300, Train Loss:  0.068, Train Acc:  96.00%, Val Loss:    0.9, Val Acc:  74.32%, Time: 0:01:03
Epoch: 13
Iter:   2400, Train Loss:  0.063, Train Acc:  98.00%, Val Loss:   0.96, Val Acc:  73.10%, Time: 0:01:06
Epoch: 14
Iter:   2500, Train Loss:  0.059, Train Acc:  98.00%, Val Loss:    1.0, Val Acc:  73.38%, Time: 0:01:09
Iter:   2600, Train Loss:  0.057, Train Acc: 100.00%, Val Loss:   0.97, Val Acc:  74.88%, Time: 0:01:11 *
Epoch: 15
Iter:   2700, Train Loss:  0.046, Train Acc: 100.00%, Val Loss:    1.0, Val Acc:  74.51%, Time: 0:01:14
Iter:   2800, Train Loss:  0.046, Train Acc:  98.00%, Val Loss:    1.1, Val Acc:  74.04%, Time: 0:01:17
Epoch: 16
Iter:   2900, Train Loss:   0.04, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  73.76%, Time: 0:01:19
Iter:   3000, Train Loss:  0.045, Train Acc:  98.00%, Val Loss:    1.1, Val Acc:  73.29%, Time: 0:01:22
Epoch: 17
Iter:   3100, Train Loss:  0.041, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  74.51%, Time: 0:01:24
Iter:   3200, Train Loss:  0.031, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  74.88%, Time: 0:01:27
Epoch: 18
Iter:   3300, Train Loss:  0.037, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  75.63%, Time: 0:01:30 *
Iter:   3400, Train Loss:  0.028, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  75.73%, Time: 0:01:32 *
Epoch: 19
Iter:   3500, Train Loss:  0.031, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  74.70%, Time: 0:01:35
Iter:   3600, Train Loss:  0.021, Train Acc:  98.00%, Val Loss:    1.2, Val Acc:  75.63%, Time: 0:01:38
Epoch: 20
Iter:   3700, Train Loss:  0.024, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  76.19%, Time: 0:01:40 *
Iter:   3800, Train Loss:  0.027, Train Acc:  98.00%, Val Loss:    1.2, Val Acc:  75.35%, Time: 0:01:43
Testing...
Test Loss:    1.2, Test Acc:  76.19%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

        POS       0.76      0.77      0.77       540
        NEG       0.76      0.76      0.76       527

avg / total       0.76      0.76      0.76      1067

Confusion Matrix...
[[415 125]
 [129 398]]
Time usage: 0:00:00
```
