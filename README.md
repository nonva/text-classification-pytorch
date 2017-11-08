# text-classification-pytorch
cnn and rnn for chinese text classification using pytorch


### CNN

#### Training

```
Configuring CNN model...
TextCNN (
  (embedding): Embedding(5000, 64)
  (conv): Conv1d(64, 256, kernel_size=(5,), stride=(1,))
  (fc1): Linear (256 -> 128)
  (fc2): Linear (128 -> 10)
)
Loading training and validation data...
Time usage: 0:00:12
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  13.28%, Val Loss:    2.4, Val Acc:  10.00%, Time: 0:00:14 *
Iter:    100, Train Loss:   0.38, Train Acc:  90.62%, Val Loss:   0.79, Val Acc:  79.10%, Time: 0:00:20 *
Iter:    200, Train Loss:   0.23, Train Acc:  93.75%, Val Loss:   0.48, Val Acc:  83.52%, Time: 0:00:27 *
Iter:    300, Train Loss:   0.33, Train Acc:  88.28%, Val Loss:   0.38, Val Acc:  86.28%, Time: 0:00:34 *
Epoch: 2
Iter:    400, Train Loss:   0.15, Train Acc:  96.09%, Val Loss:   0.29, Val Acc:  89.74%, Time: 0:00:41 *
Iter:    500, Train Loss:   0.12, Train Acc:  96.88%, Val Loss:   0.37, Val Acc:  87.30%, Time: 0:00:48
Iter:    600, Train Loss:   0.13, Train Acc:  95.31%, Val Loss:   0.28, Val Acc:  90.22%, Time: 0:00:55 *
Iter:    700, Train Loss:   0.11, Train Acc:  96.09%, Val Loss:   0.23, Val Acc:  91.36%, Time: 0:01:02 *
Epoch: 3
Iter:    800, Train Loss:  0.069, Train Acc:  96.88%, Val Loss:   0.22, Val Acc:  92.90%, Time: 0:01:09 *
Iter:    900, Train Loss:   0.11, Train Acc:  96.09%, Val Loss:   0.21, Val Acc:  93.46%, Time: 0:01:15 *
Iter:   1000, Train Loss:  0.066, Train Acc:  96.88%, Val Loss:   0.25, Val Acc:  92.10%, Time: 0:01:22
Iter:   1100, Train Loss:   0.06, Train Acc:  98.44%, Val Loss:    0.4, Val Acc:  87.34%, Time: 0:01:29
Epoch: 4
Iter:   1200, Train Loss:  0.039, Train Acc:  98.44%, Val Loss:   0.19, Val Acc:  93.46%, Time: 0:01:36
Iter:   1300, Train Loss:  0.044, Train Acc:  97.66%, Val Loss:   0.44, Val Acc:  86.18%, Time: 0:01:43
Iter:   1400, Train Loss:  0.038, Train Acc:  99.22%, Val Loss:   0.41, Val Acc:  88.06%, Time: 0:01:50
Iter:   1500, Train Loss:   0.14, Train Acc:  96.88%, Val Loss:    0.3, Val Acc:  90.18%, Time: 0:01:57
Epoch: 5
Iter:   1600, Train Loss:  0.059, Train Acc:  98.44%, Val Loss:   0.35, Val Acc:  90.48%, Time: 0:02:04
Iter:   1700, Train Loss:   0.05, Train Acc:  98.44%, Val Loss:   0.24, Val Acc:  93.38%, Time: 0:02:10
Iter:   1800, Train Loss:  0.011, Train Acc: 100.00%, Val Loss:   0.25, Val Acc:  92.08%, Time: 0:02:17
Iter:   1900, Train Loss:  0.064, Train Acc:  97.66%, Val Loss:   0.19, Val Acc:  94.62%, Time: 0:02:24 *
Epoch: 6
Iter:   2000, Train Loss:  0.058, Train Acc:  97.66%, Val Loss:    0.6, Val Acc:  86.88%, Time: 0:02:31
Iter:   2100, Train Loss:  0.016, Train Acc: 100.00%, Val Loss:   0.33, Val Acc:  90.58%, Time: 0:02:38
Iter:   2200, Train Loss: 0.0092, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  94.08%, Time: 0:02:45
Iter:   2300, Train Loss:   0.05, Train Acc:  99.22%, Val Loss:   0.23, Val Acc:  93.50%, Time: 0:02:52
Epoch: 7
Iter:   2400, Train Loss: 0.0091, Train Acc: 100.00%, Val Loss:   0.29, Val Acc:  91.88%, Time: 0:02:59
Iter:   2500, Train Loss:  0.014, Train Acc:  99.22%, Val Loss:   0.33, Val Acc:  91.42%, Time: 0:03:05
Iter:   2600, Train Loss:  0.018, Train Acc: 100.00%, Val Loss:   0.27, Val Acc:  93.98%, Time: 0:03:12
Iter:   2700, Train Loss:  0.033, Train Acc:  99.22%, Val Loss:   0.38, Val Acc:  90.34%, Time: 0:03:19
Epoch: 8
Iter:   2800, Train Loss:  0.005, Train Acc: 100.00%, Val Loss:   0.28, Val Acc:  93.00%, Time: 0:03:26
Iter:   2900, Train Loss:  0.015, Train Acc:  99.22%, Val Loss:   0.29, Val Acc:  92.56%, Time: 0:03:33
No optimization for a long time, auto-stopping...
```

#### Testing

```
Configuring CNN model...
TextCNN (
  (embedding): Embedding(5000, 64)
  (conv): Conv1d(64, 256, kernel_size=(5,), stride=(1,))
  (fc1): Linear (256 -> 128)
  (fc2): Linear (128 -> 10)
)
Loading test data...
Testing...
Test Loss:    0.2, Test Acc:  94.65%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.94      0.98      0.96      1000
         房产       0.86      0.94      0.89      1000
         家居       0.93      0.88      0.91      1000
         教育       0.90      0.94      0.92      1000
         科技       0.99      0.91      0.95      1000
         时尚       0.94      0.98      0.96      1000
         时政       0.95      0.94      0.94      1000
         游戏       0.98      0.93      0.96      1000
         娱乐       0.98      0.97      0.98      1000

avg / total       0.95      0.95      0.95     10000

Confusion Matrix...
[[995   1   0   1   2   0   0   1   0   0]
 [  0 979  15   1   2   0   0   3   0   0]
 [  1  17 935  13  10   0   2  18   2   2]
 [  2   7  85 883   9   0   8   5   0   1]
 [  1   4  13   5 942   2  11  18   2   2]
 [  0  13   3  23  19 913  19   2   5   3]
 [  1   0   0   7  10   0 981   0   1   0]
 [  0   9  28   2  20   3   0 935   1   2]
 [  0   8   9   2  23   1  16   2 934   5]
 [  0   1   3   8   9   0   5   1   5 968]]
Time usage: 0:00:04
```

### RNN

#### Training

```
Configuring RNN model...
TextRNN (
  (embedding): Embedding(5000, 64)
  (rnn): LSTM(64, 128, num_layers=2, dropout=0.2)
  (fc1): Linear (128 -> 128)
  (fc2): Linear (128 -> 10)
)
Loading training and validation data...
Time usage: 0:00:12
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  12.50%, Val Loss:    2.3, Val Acc:  10.04%, Time: 0:00:13 *
Iter:    100, Train Loss:    1.9, Train Acc:  29.69%, Val Loss:    2.1, Val Acc:  23.06%, Time: 0:00:24 *
Iter:    200, Train Loss:    1.7, Train Acc:  34.38%, Val Loss:    2.0, Val Acc:  24.22%, Time: 0:00:35 *
Iter:    300, Train Loss:    1.8, Train Acc:  29.69%, Val Loss:    2.1, Val Acc:  26.76%, Time: 0:00:45 *
Iter:    400, Train Loss:    1.7, Train Acc:  35.94%, Val Loss:    1.9, Val Acc:  27.98%, Time: 0:00:56 *
Iter:    500, Train Loss:    1.8, Train Acc:  28.12%, Val Loss:    1.9, Val Acc:  33.96%, Time: 0:01:06 *
Iter:    600, Train Loss:    1.6, Train Acc:  31.25%, Val Loss:    1.8, Val Acc:  36.04%, Time: 0:01:16 *
Iter:    700, Train Loss:    1.6, Train Acc:  46.88%, Val Loss:    1.9, Val Acc:  40.70%, Time: 0:01:27 *
Epoch: 2
Iter:    800, Train Loss:    1.6, Train Acc:  48.44%, Val Loss:    1.7, Val Acc:  39.02%, Time: 0:01:37
Iter:    900, Train Loss:    1.8, Train Acc:  28.12%, Val Loss:    1.9, Val Acc:  27.72%, Time: 0:01:48
Iter:   1000, Train Loss:    2.3, Train Acc:  12.50%, Val Loss:    2.3, Val Acc:  15.22%, Time: 0:01:58
Iter:   1100, Train Loss:    2.1, Train Acc:  26.56%, Val Loss:    2.2, Val Acc:  26.30%, Time: 0:02:09
Iter:   1200, Train Loss:    1.9, Train Acc:  28.12%, Val Loss:    2.1, Val Acc:  26.84%, Time: 0:02:19
Iter:   1300, Train Loss:    1.7, Train Acc:  40.62%, Val Loss:    1.9, Val Acc:  32.22%, Time: 0:02:30
Iter:   1400, Train Loss:    1.3, Train Acc:  56.25%, Val Loss:    1.7, Val Acc:  38.44%, Time: 0:02:41
Iter:   1500, Train Loss:    1.3, Train Acc:  50.00%, Val Loss:    1.7, Val Acc:  36.24%, Time: 0:02:51
Epoch: 3
Iter:   1600, Train Loss:    1.5, Train Acc:  42.19%, Val Loss:    1.7, Val Acc:  40.72%, Time: 0:03:02 *
Iter:   1700, Train Loss:   0.89, Train Acc:  71.88%, Val Loss:    1.3, Val Acc:  56.76%, Time: 0:03:12 *
Iter:   1800, Train Loss:    0.9, Train Acc:  60.94%, Val Loss:    1.3, Val Acc:  53.00%, Time: 0:03:23
Iter:   1900, Train Loss:   0.81, Train Acc:  68.75%, Val Loss:    1.0, Val Acc:  62.56%, Time: 0:03:33 *
Iter:   2000, Train Loss:   0.74, Train Acc:  73.44%, Val Loss:    1.0, Val Acc:  68.58%, Time: 0:03:44 *
Iter:   2100, Train Loss:   0.61, Train Acc:  78.12%, Val Loss:   0.81, Val Acc:  74.76%, Time: 0:03:54 *
Iter:   2200, Train Loss:   0.59, Train Acc:  79.69%, Val Loss:   0.83, Val Acc:  74.78%, Time: 0:04:05 *
Iter:   2300, Train Loss:   0.63, Train Acc:  79.69%, Val Loss:   0.76, Val Acc:  77.22%, Time: 0:04:15 *
Epoch: 4
Iter:   2400, Train Loss:   0.41, Train Acc:  84.38%, Val Loss:   0.74, Val Acc:  78.82%, Time: 0:04:26 *
Iter:   2500, Train Loss:   0.41, Train Acc:  84.38%, Val Loss:   0.59, Val Acc:  83.08%, Time: 0:04:36 *
Iter:   2600, Train Loss:   0.55, Train Acc:  87.50%, Val Loss:   0.62, Val Acc:  81.48%, Time: 0:04:47
Iter:   2700, Train Loss:   0.22, Train Acc:  92.19%, Val Loss:   0.57, Val Acc:  84.24%, Time: 0:04:58 *
Iter:   2800, Train Loss:   0.43, Train Acc:  90.62%, Val Loss:   0.54, Val Acc:  83.64%, Time: 0:05:08
Iter:   2900, Train Loss:   0.27, Train Acc:  95.31%, Val Loss:   0.44, Val Acc:  86.94%, Time: 0:05:19 *
Iter:   3000, Train Loss:   0.37, Train Acc:  89.06%, Val Loss:   0.44, Val Acc:  86.70%, Time: 0:05:29
Iter:   3100, Train Loss:   0.16, Train Acc:  96.88%, Val Loss:   0.49, Val Acc:  84.74%, Time: 0:05:40
Epoch: 5
Iter:   3200, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:   0.46, Val Acc:  86.00%, Time: 0:05:50
Iter:   3300, Train Loss:   0.31, Train Acc:  89.06%, Val Loss:    0.4, Val Acc:  88.08%, Time: 0:06:01 *
Iter:   3400, Train Loss:   0.12, Train Acc:  95.31%, Val Loss:    0.4, Val Acc:  87.98%, Time: 0:06:11
Iter:   3500, Train Loss:   0.29, Train Acc:  93.75%, Val Loss:    0.5, Val Acc:  83.90%, Time: 0:06:22
Iter:   3600, Train Loss:   0.33, Train Acc:  87.50%, Val Loss:   0.39, Val Acc:  87.74%, Time: 0:06:32
Iter:   3700, Train Loss:   0.26, Train Acc:  95.31%, Val Loss:   0.31, Val Acc:  90.52%, Time: 0:06:43 *
Iter:   3800, Train Loss:   0.16, Train Acc:  93.75%, Val Loss:   0.45, Val Acc:  85.92%, Time: 0:06:53
Iter:   3900, Train Loss:   0.19, Train Acc:  93.75%, Val Loss:   0.36, Val Acc:  88.50%, Time: 0:07:04
Epoch: 6
Iter:   4000, Train Loss:   0.42, Train Acc:  87.50%, Val Loss:   0.35, Val Acc:  88.68%, Time: 0:07:14
Iter:   4100, Train Loss:   0.24, Train Acc:  92.19%, Val Loss:   0.36, Val Acc:  88.46%, Time: 0:07:25
Iter:   4200, Train Loss:   0.32, Train Acc:  90.62%, Val Loss:    0.5, Val Acc:  85.10%, Time: 0:07:35
Iter:   4300, Train Loss:   0.33, Train Acc:  90.62%, Val Loss:   0.39, Val Acc:  87.22%, Time: 0:07:46
Iter:   4400, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:    0.3, Val Acc:  90.58%, Time: 0:07:56 *
Iter:   4500, Train Loss:   0.17, Train Acc:  92.19%, Val Loss:   0.37, Val Acc:  88.50%, Time: 0:08:07
Iter:   4600, Train Loss:    0.3, Train Acc:  89.06%, Val Loss:   0.53, Val Acc:  83.66%, Time: 0:08:17
Epoch: 7
Iter:   4700, Train Loss:  0.072, Train Acc:  96.88%, Val Loss:   0.35, Val Acc:  88.34%, Time: 0:08:27
Iter:   4800, Train Loss:   0.23, Train Acc:  90.62%, Val Loss:   0.41, Val Acc:  87.74%, Time: 0:08:38
Iter:   4900, Train Loss:   0.12, Train Acc:  96.88%, Val Loss:   0.31, Val Acc:  90.34%, Time: 0:08:48
Iter:   5000, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:    0.3, Val Acc:  91.02%, Time: 0:08:59 *
Iter:   5100, Train Loss:   0.22, Train Acc:  93.75%, Val Loss:   0.28, Val Acc:  90.82%, Time: 0:09:09
Iter:   5200, Train Loss:   0.11, Train Acc:  95.31%, Val Loss:   0.36, Val Acc:  88.54%, Time: 0:09:20
Iter:   5300, Train Loss:   0.25, Train Acc:  92.19%, Val Loss:   0.28, Val Acc:  91.08%, Time: 0:09:30 *
Iter:   5400, Train Loss:    0.1, Train Acc:  95.31%, Val Loss:   0.34, Val Acc:  89.36%, Time: 0:09:41
Epoch: 8
Iter:   5500, Train Loss:  0.076, Train Acc:  98.44%, Val Loss:   0.34, Val Acc:  89.56%, Time: 0:09:51
Iter:   5600, Train Loss:   0.22, Train Acc:  92.19%, Val Loss:   0.29, Val Acc:  91.40%, Time: 0:10:02 *
Iter:   5700, Train Loss:  0.073, Train Acc: 100.00%, Val Loss:   0.29, Val Acc:  91.28%, Time: 0:10:12
Iter:   5800, Train Loss:    0.2, Train Acc:  92.19%, Val Loss:   0.38, Val Acc:  88.28%, Time: 0:10:23
Iter:   5900, Train Loss:  0.093, Train Acc:  96.88%, Val Loss:   0.27, Val Acc:  91.60%, Time: 0:10:33 *
Iter:   6000, Train Loss:    0.1, Train Acc:  98.44%, Val Loss:   0.27, Val Acc:  91.64%, Time: 0:10:44 *
Iter:   6100, Train Loss:   0.18, Train Acc:  93.75%, Val Loss:   0.27, Val Acc:  91.66%, Time: 0:10:54 *
Iter:   6200, Train Loss:  0.026, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  92.80%, Time: 0:11:05 *
Epoch: 9
Iter:   6300, Train Loss:   0.25, Train Acc:  92.19%, Val Loss:    0.4, Val Acc:  88.02%, Time: 0:11:15
Iter:   6400, Train Loss:   0.12, Train Acc:  96.88%, Val Loss:   0.33, Val Acc:  89.10%, Time: 0:11:26
Iter:   6500, Train Loss:  0.093, Train Acc:  96.88%, Val Loss:   0.24, Val Acc:  93.46%, Time: 0:11:36 *
Iter:   6600, Train Loss:   0.14, Train Acc:  95.31%, Val Loss:   0.36, Val Acc:  88.90%, Time: 0:11:47
Iter:   6700, Train Loss:  0.064, Train Acc:  98.44%, Val Loss:   0.29, Val Acc:  90.34%, Time: 0:11:57
Iter:   6800, Train Loss:   0.26, Train Acc:  93.75%, Val Loss:   0.25, Val Acc:  92.70%, Time: 0:12:07
Iter:   6900, Train Loss:   0.18, Train Acc:  93.75%, Val Loss:   0.29, Val Acc:  90.98%, Time: 0:12:18
Iter:   7000, Train Loss:  0.067, Train Acc:  98.44%, Val Loss:   0.41, Val Acc:  87.84%, Time: 0:12:28
Epoch: 10
Iter:   7100, Train Loss:   0.14, Train Acc:  95.31%, Val Loss:   0.41, Val Acc:  88.64%, Time: 0:12:39
Iter:   7200, Train Loss:    0.2, Train Acc:  95.31%, Val Loss:   0.25, Val Acc:  92.50%, Time: 0:12:49
Iter:   7300, Train Loss:   0.28, Train Acc:  89.06%, Val Loss:   0.36, Val Acc:  90.16%, Time: 0:13:00
Iter:   7400, Train Loss:   0.13, Train Acc:  96.88%, Val Loss:   0.27, Val Acc:  92.12%, Time: 0:13:10
Iter:   7500, Train Loss:   0.14, Train Acc:  98.44%, Val Loss:   0.29, Val Acc:  90.90%, Time: 0:13:21
No optimization for a long time, auto-stopping...
```

#### Testing

```
Configuring RNN model...
TextRNN (
  (embedding): Embedding(5000, 64)
  (rnn): LSTM(64, 128, num_layers=2, dropout=0.2)
  (fc1): Linear (128 -> 128)
  (fc2): Linear (128 -> 10)
)
Loading test data...
Testing...
Test Loss:   0.23, Test Acc:  93.72%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.90      0.99      0.94      1000
         房产       0.91      0.88      0.89      1000
         家居       0.91      0.81      0.86      1000
         教育       0.89      0.92      0.91      1000
         科技       0.95      0.95      0.95      1000
         时尚       0.94      0.96      0.95      1000
         时政       0.92      0.95      0.93      1000
         游戏       0.98      0.94      0.96      1000
         娱乐       0.98      0.98      0.98      1000

avg / total       0.94      0.94      0.94     10000

Confusion Matrix...
[[995   2   0   0   0   0   0   2   0   1]
 [  0 989   3   1   3   0   0   4   0   0]
 [  1  45 881  18  18   1   3  32   0   1]
 [  4  24  67 807  50   7  16  18   1   6]
 [  3   9   4  15 924  20   4  18   1   2]
 [  0   7   1   9  13 952   4   8   6   0]
 [  0   0   0  28   7   1 962   0   2   0]
 [  0  15  13   3  16   4   0 947   2   0]
 [  1   3   1   1   8   8  26   0 939  13]
 [  1   3   2   4   2   4   3   2   3 976]]
Time usage: 0:00:05
```



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
