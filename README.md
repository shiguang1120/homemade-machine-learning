# 自制机器学习

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trekhleb/homemade-machine-learning/master?filepath=notebooks)
[![Build Status](https://travis-ci.org/trekhleb/homemade-machine-learning.svg?branch=master)](https://travis-ci.org/trekhleb/homemade-machine-learning)

_对于此存储库的Octave/MatLab版本，请查看 [machine-learning-octave](https://github.com/shiguang1120/machine-learning-octave) 项目。_

> 该存储库包含在** Python **中实现的流行机器学习算法的示例，其背后的数学被解释。 每个算法都有交互式** Jupyter Notebook **演示，允许您使用训练数据，算法配置，并立即在浏览器中查看结果，图表和预测**。 在大多数情况下，解释是基于Andrew Ng的[这个伟大的机器学习课程](https://www.coursera.org/learn/machine-learning)。

这个存储库的目的是通过使用3个<sup>rd</sup>方库的单行_but_来实现机器学习算法，而不是从头开始实践这些算法，并更好地理解每个算法背后的数学。 这就是为什么所有算法实现都被称为“自制”并且不打算用于生产的原因。

## 监督学习

在监督学习中，我们将一组训练数据作为输入，并将每组训练集的标签或“正确答案”作为输出。 然后我们正在训练我们的模型（机器学习算法参数）以正确地将输入映射到输出（以进行正确的预测）。 最终目的是找到这样的模型参数，即使对于新的输入示例，也能成功地继续正确_input→output_ mapping（预测）。

### 回归

在回归问题中，我们进行实际的价值预测。 基本上我们尝试沿着训练样例绘制线/平面/n维平面。

_用法示例：股票价格预测，销售分析，任意数量的依赖性等。_

#### 🤖 线性回归

- 📗 [数学| 线性回归](homemade/linear_regression) - 理论和进一步阅读的链接
- ⚙️ [代码| 线性回归](homemade/linear_regression/linear_regression.py) - 实施例
- ▶️ [演示| 单变量线性回归](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/univariate_linear_regression_demo.ipynb) - 用`economy GDP`预测`country happiness`得分
- ▶️ [演示| 多元线性回归](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/multivariate_linear_regression_demo.ipynb) - 用`economy GDP`和`freedom index` 来预测`country happiness`得分
- ▶️ [演示| 非线性回归](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/non_linear_regression_demo.ipynb) - 使用带有_polynomial_和_sinusoid_特征的线性回归来预测非线性依赖关系

### 分类

在分类问题中，我们通过某些特征分割输入示例。

_用法示例: 垃圾邮件过滤器，语言检测，查找类似文档，手写字母识别等._

#### 🤖 Logistic回归

- 📗 [数学| Logistic回归](homemade/logistic_regression) - 理论和进一步阅读的链接
- ⚙️ [代码| Logistic回归](homemade/logistic_regression/logistic_regression.py) - 实施例
- ▶️ [演示| Logistic回归（线性边界）](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/logistic_regression_with_linear_boundary_demo.ipynb) - 根据`petal_length`和`petal_width`预测虹膜花`class`
- ▶️ [演示| Logistic回归（非线性边界）](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/logistic_regression_with_non_linear_boundary_demo.ipynb) - 根据`param_1`和`param_2`预测微芯片的`validity`
- ▶️ [演示| 多元Logistic回归|MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/multivariate_logistic_regression_demo.ipynb) - 识别来自`28x28`像素图像的手写数字
- ▶️ [演示| 多元Logistic回归|Fashion-MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/multivariate_logistic_regression_fashion_demo.ipynb) - 识别`28x28`像素图像中的衣服类型

## 无监督学习

无监督学习是机器学习的一个分支，它从未经标记，分类或分类的测试数据中学习。 无监督学习不是响应反馈，而是根据每个新数据中是否存在这种共性来识别数据中的共性并做出反应。

### 聚类

在聚类问题中，我们将训练样例分成未知特征。 算法本身决定了用于分裂的特征。

_用法示例: 市场细分，社交网络分析，组织计算集群，天文数据分析，图像压缩等_

#### 🤖 K均值算法

- 📗 [算法| K均值算法](homemade/k_means) - 理论和进一步阅读的链接
- ⚙️ [代码| K均值算法](homemade/k_means/k_means.py) - 实施例
- ▶️ [演示| K均值算法](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/k_means/k_means_demo.ipynb) - 根据`petal_length`和`petal_width`将鸢尾花分成簇

### 异常检测

异常检测（也是异常检测）是通过与大多数数据显着不同而引起怀疑的罕见项目，事件或观察的识别。

_用法示例：入侵检测，欺诈检测，系统健康监控，从数据集中删除异常数据等。_

#### 🤖 利用高斯分布进行异常检测

- 📗 [算法| 利用高斯分布进行异常检测](homemade/anomaly_detection) - 理论和进一步阅读的链接
- ⚙️ [代码| 利用高斯分布进行异常检测](homemade/anomaly_detection/gaussian_anomaly_detection.py) - 实施例
- ▶️ [演示| 异常检测](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/anomaly_detection/anomaly_detection_gaussian_demo.ipynb) - 发现服务器操作参数中的异常，例如`latency`和`threshold`

## 神经网络（NN）

神经网络本身不是算法，而是许多不同机器学习算法的框架，它们协同工作并处理复杂的数据输入。

_用法示例: 作为所有其他算法的替代，一般来说，图像识别，语音识别，图像处理（应用特定风格），语言翻译等。_

#### 🤖 多层感知器（MLP）

- 📗 [算法| 多层感知器](homemade/neural_network) - 理论和进一步阅读的链接
- ⚙️ [代码| 多层感知器](homemade/neural_network/multilayer_perceptron.py) - 实施例
- ▶️ [演示| 多层感知器|MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/neural_network/multilayer_perceptron_demo.ipynb) - 识别来自`28x28`像素图像的手写数字
- ▶️ [演示| 多层感知器|Fashion-MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/neural_network/multilayer_perceptron_fashion_demo.ipynb) - 从`28x28`像素图像识别衣服的类型

## 机器学习地图

![Machine Learning Map](images/machine-learning-map.png)

以下机器学习主题图的来源是[这篇精彩的博文](https://vas3k.ru/blog/machine_learning/)

## 先决条件

#### 安装Python

确保您的计算机上安装了[Python installed](https://realpython.com/installing-python/)。

您可能想使用[venv](https://docs.python.org/3/library/venv.html) 标准Python库
创建虚拟环境并安装Python，`pip`和所有依赖包
从本地项目目录提供，以避免搞乱系统范围的包及其
版本。

#### 安装依赖项

通过运行以下命令安装项目所需的所有依赖项：

```bash
pip install -r requirements.txt
```

#### 在本地安装Jupyter

项目中的所有演示可以直接在浏览器中运行，而无需在本地安装Jupyter。但是如果你想在本地启动[Jupyter Notebook](http://jupyter.org/)，你可以通过从项目的根文件夹运行以下命令来实现：

```bash
jupyter notebook
```
在这之后，Jupyter笔记本将被访问 `http://localhost:8888`.

#### 远程启动Jupyter

每个算法部分都包含[Jupyter NBViewer](http://nbviewer.jupyter.org/)的演示链接。这是Jupyter笔记本电脑的快速在线预览器，您可以在浏览器中查看演示代码，图表和数据，而无需在本地安装任何内容。 如果你想用演示笔记本_change_代码和_experiment_你需要在[Binder](https://mybinder.org/)中启动笔记本。 您只需单击NBViewer右上角的_“Execute on Binder”_链接即可完成此操作。

![](./images/binder-button-place.png)

## 数据集

可以在[数据文件夹](data)中找到用于Jupyter Notebook演示的数据集列表。
