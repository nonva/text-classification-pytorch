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
Iter:    100, Train Loss:   0.76, Train Acc:  60.00%, Val Loss:   0.67, Val Acc:  57.83%, Time: 0:00:05 *
Epoch: 2
Iter:    200, Train Loss:   0.69, Train Acc:  66.00%, Val Loss:   0.65, Val Acc:  61.01%, Time: 0:00:08 *
Iter:    300, Train Loss:   0.61, Train Acc:  60.00%, Val Loss:   0.65, Val Acc:  60.07%, Time: 0:00:11
Epoch: 3
Iter:    400, Train Loss:   0.59, Train Acc:  60.00%, Val Loss:   0.63, Val Acc:  62.23%, Time: 0:00:13 *
Iter:    500, Train Loss:   0.52, Train Acc:  76.00%, Val Loss:   0.62, Val Acc:  66.17%, Time: 0:00:16 *
Epoch: 4
Iter:    600, Train Loss:   0.49, Train Acc:  82.00%, Val Loss:   0.59, Val Acc:  69.45%, Time: 0:00:19 *
Iter:    700, Train Loss:   0.43, Train Acc:  78.00%, Val Loss:   0.58, Val Acc:  71.04%, Time: 0:00:21 *
Epoch: 5
Iter:    800, Train Loss:    0.4, Train Acc:  80.00%, Val Loss:    0.6, Val Acc:  70.57%, Time: 0:00:24
Iter:    900, Train Loss:   0.32, Train Acc:  84.00%, Val Loss:    0.6, Val Acc:  71.60%, Time: 0:00:26 *
Epoch: 6
Iter:   1000, Train Loss:    0.3, Train Acc:  82.00%, Val Loss:   0.64, Val Acc:  70.67%, Time: 0:00:29
Iter:   1100, Train Loss:   0.26, Train Acc:  94.00%, Val Loss:   0.62, Val Acc:  72.16%, Time: 0:00:32 *
Epoch: 7
Iter:   1200, Train Loss:   0.23, Train Acc:  90.00%, Val Loss:   0.66, Val Acc:  72.63%, Time: 0:00:34 *
Iter:   1300, Train Loss:    0.2, Train Acc:  96.00%, Val Loss:   0.71, Val Acc:  70.67%, Time: 0:00:37
Epoch: 8
Iter:   1400, Train Loss:   0.18, Train Acc:  94.00%, Val Loss:    0.7, Val Acc:  73.57%, Time: 0:00:40 *
Iter:   1500, Train Loss:   0.14, Train Acc:  84.00%, Val Loss:   0.78, Val Acc:  70.57%, Time: 0:00:42
Epoch: 9
Iter:   1600, Train Loss:   0.15, Train Acc:  96.00%, Val Loss:   0.74, Val Acc:  72.91%, Time: 0:00:45
Iter:   1700, Train Loss:   0.13, Train Acc:  94.00%, Val Loss:   0.85, Val Acc:  73.57%, Time: 0:00:48
Epoch: 10
Iter:   1800, Train Loss:   0.13, Train Acc:  98.00%, Val Loss:   0.81, Val Acc:  73.01%, Time: 0:00:50
Iter:   1900, Train Loss:  0.095, Train Acc:  94.00%, Val Loss:    1.0, Val Acc:  71.98%, Time: 0:00:53
Epoch: 11
Iter:   2000, Train Loss:  0.097, Train Acc:  98.00%, Val Loss:   0.84, Val Acc:  73.48%, Time: 0:00:56
Iter:   2100, Train Loss:  0.078, Train Acc:  98.00%, Val Loss:   0.86, Val Acc:  74.04%, Time: 0:00:58 *
Epoch: 12
Iter:   2200, Train Loss:  0.074, Train Acc:  94.00%, Val Loss:   0.91, Val Acc:  73.66%, Time: 0:01:01
Iter:   2300, Train Loss:  0.073, Train Acc: 100.00%, Val Loss:   0.98, Val Acc:  71.51%, Time: 0:01:04
Epoch: 13
Iter:   2400, Train Loss:  0.063, Train Acc:  98.00%, Val Loss:   0.96, Val Acc:  74.51%, Time: 0:01:06 *
Epoch: 14
Iter:   2500, Train Loss:  0.058, Train Acc:  90.00%, Val Loss:    1.0, Val Acc:  72.35%, Time: 0:01:09
Iter:   2600, Train Loss:  0.049, Train Acc:  98.00%, Val Loss:    1.0, Val Acc:  74.79%, Time: 0:01:12 *
Epoch: 15
Iter:   2700, Train Loss:   0.04, Train Acc:  98.00%, Val Loss:    1.0, Val Acc:  73.66%, Time: 0:01:14
Iter:   2800, Train Loss:  0.042, Train Acc:  98.00%, Val Loss:    1.1, Val Acc:  74.13%, Time: 0:01:17
Epoch: 16
Iter:   2900, Train Loss:  0.039, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  73.20%, Time: 0:01:20
Iter:   3000, Train Loss:  0.042, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  74.70%, Time: 0:01:22
Epoch: 17
Iter:   3100, Train Loss:  0.032, Train Acc:  98.00%, Val Loss:    1.2, Val Acc:  75.35%, Time: 0:01:25 *
Iter:   3200, Train Loss:  0.037, Train Acc:  98.00%, Val Loss:    1.2, Val Acc:  74.13%, Time: 0:01:27
Epoch: 18
Iter:   3300, Train Loss:  0.034, Train Acc:  96.00%, Val Loss:    1.2, Val Acc:  74.98%, Time: 0:01:30
Iter:   3400, Train Loss:  0.034, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  75.26%, Time: 0:01:33
Epoch: 19
Iter:   3500, Train Loss:  0.035, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  74.79%, Time: 0:01:36
Iter:   3600, Train Loss:  0.041, Train Acc: 100.00%, Val Loss:    1.3, Val Acc:  74.60%, Time: 0:01:38
Epoch: 20
Iter:   3700, Train Loss:  0.035, Train Acc:  98.00%, Val Loss:    1.3, Val Acc:  75.26%, Time: 0:01:41
Iter:   3800, Train Loss:  0.033, Train Acc:  96.00%, Val Loss:    1.4, Val Acc:  74.88%, Time: 0:01:43
Testing...
Test Loss:    1.2, Test Acc:  75.35%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

        POS       0.73      0.79      0.76       527
        NEG       0.78      0.72      0.75       540

avg / total       0.76      0.75      0.75      1067

Confusion Matrix...
[[416 111]
 [152 388]]
Time usage: 0:00:00
```
