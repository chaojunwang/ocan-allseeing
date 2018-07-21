# ocan-allseeing
ocan with kaggle data

### 1.data_preprocessing
#### 数据预处理：非数值型原始特征 -> categorical -> one-hot 

### 2.DNN_Autoencoder
#### one-hot编码的categorical features作为输入进行encoding，映射到另一维度得到一种好的特征表示

### 3.OCAN_model
#### 使用one-class gan 生成互补数据，以训练分类器
