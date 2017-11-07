# text-classification-pytorch
cnn and rnn for chinese text classification using pytorch


### CNN

#### Training


#### Testing

```
Configuring CNN model...
TextCNN (
  (embedding): Embedding(5000, 64)
  (conv): Conv1d(64, 64, kernel_size=(5,), stride=(1,))
  (fc1): Linear (64 -> 128)
  (fc2): Linear (128 -> 10)
)
Loading test data...
Testing...
Test Loss:   0.33, Test Acc:  91.26%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.89      0.97      0.93      1000
         房产       0.84      0.81      0.83      1000
         家居       0.90      0.74      0.81      1000
         教育       0.91      0.85      0.88      1000
         科技       0.86      0.96      0.91      1000
         时尚       0.97      0.96      0.97      1000
         时政       0.87      0.93      0.90      1000
         游戏       0.95      0.93      0.94      1000
         娱乐       0.96      0.97      0.96      1000

avg / total       0.91      0.91      0.91     10000

Confusion Matrix...
[[994   1   0   1   2   0   0   0   1   1]
 [  0 972   9   2   1   2   1   9   3   1]
 [  3  56 815  27  12   9   2  67   3   6]
 [  5  30 110 741  29  37   9  29   5   5]
 [  1  13   6  10 846  63   3  32  22   4]
 [  0   5   1  12   6 964   2   0   7   3]
 [  4   2   0  16   6   4 960   1   6   1]
 [  0  11  22   0  12  19   0 930   3   3]
 [  0   4   6   2  15  11   8   3 934  17]
 [  2   2   1  10   4   6   1   0   4 970]]
Time usage: 0:00:05
```