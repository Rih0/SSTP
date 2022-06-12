## 1 描述(Description)
模型(SSTP)的实现。
## 2 构建数据集(Build Dataset)
数据以文本形式(.txt)存储，每行数据如下格式表示：

`user_id  poi_id  latitude  longitude timestamp day_id  category_id UTCTime`
## 3 运行(Run)
`python main.py --dataset NYC`
## 4 环境(Env)
dgl-cuda11.1	0.7.2

pytorch	1.7.1

python	3.7.11
