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
  (conv13): Conv1d(128, 128, kernel_size=(3,), stride=(1,))
  (conv14): Conv1d(128, 128, kernel_size=(4,), stride=(1,))
  (conv15): Conv1d(128, 128, kernel_size=(5,), stride=(1,))
  (fc1): Linear (384 -> 2)
  (dropout): Dropout (p = 0.5)
)
Epoch: 1
Iter:    100, Train Loss:   0.74, Train Acc:  48.44%, Val Loss:   0.65, Val Acc:  62.42%, Time: 0:00:05 *
Epoch: 2
Iter:    200, Train Loss:   0.65, Train Acc:  60.94%, Val Loss:   0.67, Val Acc:  60.54%, Time: 0:00:08
Iter:    300, Train Loss:   0.59, Train Acc:  74.58%, Val Loss:   0.64, Val Acc:  62.98%, Time: 0:00:11 *
Epoch: 3
Iter:    400, Train Loss:   0.51, Train Acc:  75.00%, Val Loss:   0.59, Val Acc:  68.98%, Time: 0:00:14 *
Epoch: 4
Iter:    500, Train Loss:   0.46, Train Acc:  87.50%, Val Loss:   0.61, Val Acc:  68.13%, Time: 0:00:16
Iter:    600, Train Loss:   0.39, Train Acc:  77.97%, Val Loss:   0.58, Val Acc:  70.67%, Time: 0:00:19 *
Epoch: 5
Iter:    700, Train Loss:   0.34, Train Acc:  73.44%, Val Loss:   0.63, Val Acc:  69.17%, Time: 0:00:22
Epoch: 6
Iter:    800, Train Loss:   0.28, Train Acc:  92.19%, Val Loss:   0.61, Val Acc:  72.07%, Time: 0:00:25 *
Iter:    900, Train Loss:   0.24, Train Acc:  89.83%, Val Loss:   0.64, Val Acc:  72.26%, Time: 0:00:27 *
Epoch: 7
Iter:   1000, Train Loss:    0.2, Train Acc:  87.50%, Val Loss:   0.66, Val Acc:  72.45%, Time: 0:00:30 *
Epoch: 8
Iter:   1100, Train Loss:   0.17, Train Acc:  96.88%, Val Loss:   0.68, Val Acc:  72.16%, Time: 0:00:33
Iter:   1200, Train Loss:   0.17, Train Acc:  98.31%, Val Loss:    0.7, Val Acc:  74.32%, Time: 0:00:36 *
Epoch: 9
Iter:   1300, Train Loss:   0.13, Train Acc:  93.75%, Val Loss:    0.9, Val Acc:  71.04%, Time: 0:00:39
Epoch: 10
Iter:   1400, Train Loss:   0.12, Train Acc: 100.00%, Val Loss:   0.75, Val Acc:  73.38%, Time: 0:00:41
Iter:   1500, Train Loss:   0.11, Train Acc: 100.00%, Val Loss:   0.83, Val Acc:  73.85%, Time: 0:00:44
Epoch: 11
Iter:   1600, Train Loss:  0.093, Train Acc:  96.88%, Val Loss:   0.83, Val Acc:  74.23%, Time: 0:00:47
Epoch: 12
Iter:   1700, Train Loss:  0.082, Train Acc:  96.88%, Val Loss:   0.86, Val Acc:  74.79%, Time: 0:00:50 *
Iter:   1800, Train Loss:  0.064, Train Acc: 100.00%, Val Loss:   0.94, Val Acc:  72.73%, Time: 0:00:53
Epoch: 13
Iter:   1900, Train Loss:  0.052, Train Acc:  98.44%, Val Loss:   0.94, Val Acc:  73.48%, Time: 0:00:55
Epoch: 14
Iter:   2000, Train Loss:  0.051, Train Acc:  95.31%, Val Loss:    1.0, Val Acc:  74.41%, Time: 0:00:58
Iter:   2100, Train Loss:  0.039, Train Acc: 100.00%, Val Loss:   0.99, Val Acc:  74.79%, Time: 0:01:01
Epoch: 15
Iter:   2200, Train Loss:  0.042, Train Acc: 100.00%, Val Loss:    1.0, Val Acc:  74.13%, Time: 0:01:04
Epoch: 16
Iter:   2300, Train Loss:  0.036, Train Acc: 100.00%, Val Loss:    1.0, Val Acc:  75.63%, Time: 0:01:06 *
Iter:   2400, Train Loss:  0.036, Train Acc:  98.31%, Val Loss:    1.1, Val Acc:  75.16%, Time: 0:01:09
Epoch: 17
Iter:   2500, Train Loss:  0.035, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  74.98%, Time: 0:01:12
Epoch: 18
Iter:   2600, Train Loss:  0.029, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  74.60%, Time: 0:01:15
Iter:   2700, Train Loss:  0.028, Train Acc: 100.00%, Val Loss:    1.1, Val Acc:  74.51%, Time: 0:01:17
Epoch: 19
Iter:   2800, Train Loss:  0.033, Train Acc:  98.44%, Val Loss:    1.2, Val Acc:  74.41%, Time: 0:01:20
Epoch: 20
Iter:   2900, Train Loss:   0.03, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  74.98%, Time: 0:01:23
Iter:   3000, Train Loss:  0.032, Train Acc: 100.00%, Val Loss:    1.2, Val Acc:  74.98%, Time: 0:01:26
Testing...
Test Loss:    1.0, Test Acc:  75.63%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

        POS       0.75      0.75      0.75       515
        NEG       0.76      0.77      0.76       552

avg / total       0.76      0.76      0.76      1067

Confusion Matrix...
[[384 131]
 [129 423]]
Time usage: 0:00:00
```
