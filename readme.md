配置环境：
- pandas==2.0.3

- Pillow==9.4.0

- scikit_learn==1.3.0

- torch==2.1.0

- torchvision==0.16.0

  

运行代码：

```python
cd project5/
python hw5.py
```



 文件结构：

```python
  |-- project5
    |-- dataset/  # 存放本次实验所用的数据
       |-- data/ # 包括所有的训练文本和图片，每个文件按照唯一的guid命名
       |-- train.txt # 数据的guid和对应的情感标签
       |-- test_without_label.txt # 数据的guid和空的情感标签
    |-- hw5.py # 本次实验的代码
  |-- requirements.txt # 本次实验的配置环境
  |-- results.txt # 本次实验的预测文件
  |-- readme.md 
```



