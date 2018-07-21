# ocan-allseeing
ocan with kaggle data

### 1.data_preprocessing
#### 数据预处理：非数值型原始特征 -> categorical -> one-hot 

### 2.DNN_Autoencoder
#### one-hot编码的categorical features作为输入进行encoding，映射到另一维度得到一种好的特征表示

### 3.OCAN_model
#### 使用one-class gan 生成互补数据，以训练分类器

              precision    recall  f1-score   support

     benign     0.9003    0.9472    0.9232     10000
     vandal     0.6983    0.5381    0.6078      2271

avg / total     0.8629    0.8715    0.8648     12271

acc:0.8714856164941732

### Threshold is  0.46277308
### OCAN model finds 5 vandals out of first 8 vandals.
### OCAN model finds 8 vandals out of first 12 vandals.
### OCAN model finds 14 vandals out of first 20 vandals.
### OCAN model finds 60 vandals out of first 100 vandals.
### OCAN model finds 302 vandals out of first 500 vandals.
### OCAN model finds 1363 vandals out of first 2271 vandals.
