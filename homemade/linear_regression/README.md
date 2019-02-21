# 线性回归

## Jupyter演示

▶️ [演示| 单变量线性回归](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/univariate_linear_regression_demo.ipynb) - 用`economy GDP`预测`country happiness`得分

▶️ [演示| 多元线性回归](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/multivariate_linear_regression_demo.ipynb) - 用`economy GDP`和`freedom index` 来预测`country happiness`得分

▶️ [演示| 非线性回归](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/non_linear_regression_demo.ipynb) - 使用带有_polynomial_和_sinusoid_特征的线性回归来预测非线性依赖关系

## 定义

**线性回归** 是线性模型，例如假设输入变量(_x_)和单个输出变量(_y_)之间存在线性关系的模型。更具体地，可以从输入变量(_x_)的线性组合计算输出变量(_y_)。

![Linear Regression](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)

在上图中，输入变量_x_和输出变量_y_之间存在依赖关系的示例。上图中的红线称为最佳拟合直线。基于给定的数据点（训练示例），我们尝试绘制一条最佳模拟点的线。在现实世界中，我们通常有多个输入变量。

## 功能（变量）

每个培训示例都包含描述此示例的特征（变量）（即房间数量，公寓的平方等）

![Features](../../images/linear_regression/features.svg)

_n_ - 一些特征

_R<sup>n+1</sup>_ - _n+1_个实数的向量

## 参数

假设的参数我们希望我们的算法学习以便能够做预测（即预测公寓的价格）。

![Parameters](../../images/linear_regression/parameters.svg)

## 假设

将特征和参数作为输入并将值预测为输出的等式（即根据房间的大小和数量预测公寓的价格）。

![Hypothesis](../../images/linear_regression/hypothesis.svg)

为方便起见，请定义_X<sub>0</sub> = 1_

## 成本函数

显示假设的预测与当前参数集的准确程度的函数。

![Cost Function](../../images/linear_regression/cost-function.svg)

_x<sup>i</sup>_ - 第 _i<sup>个</sup>_训练样例的输入（特征）

_y<sup>i</sup>_ - 第 _i<sup>个</sup>_训练样例的输出

_m_ - 训练样例数量

## 批量梯度下降

梯度下降是用于找到上述成本函数的最小值的迭代优化算法。为了使用梯度下降找到函数的局部最小值，需要采用与当前点处函数的梯度（或近似梯度）的负值成比例的步长。

下面的图片说明了我们从山上下来寻找当地最低限度的步骤。

![Gradient Descent](https://cdn-images-1.medium.com/max/1600/1*f9a162GhpMbiTVTAua_lLQ.png)

步骤的方向由当前点的成本函数的导数定义。

![Gradient Descent](https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png)

一旦我们决定了我们需要走的方向，我们需要决定我们需要采取的步骤大小。

![Gradient Descent](https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png)

我们需要同时更新西塔[Theta](../../images/linear_regression/theta-j.svg) for _j = 0, 1, ..., n_

![Gradient Descent](../../images/linear_regression/gradient-descent-1.svg)

![Gradient Descent](../../images/linear_regression/gradient-descent-2.svg)

![alpha](../../images/linear_regression/alpha.svg) - the learning rate, the constant that defines the size of the gradient descent step

![x-i-j](../../images/linear_regression/x-i-j.svg) - _j<sup>th</sup>_ feature value of the _i<sup>th</sup>_ training example

![x-i](../../images/linear_regression/x-i.svg) - input (features) of _i<sup>th</sup>_ training example

_y<sup>i</sup>_ - output of _i<sup>th</sup>_ training example

_m_ - number of training examples

_n_ - number of features

> When we use term "batch" for gradient descent it means that each step of gradient descent uses **all** the training examples (as you might see from the formula above).

## Feature Scaling

To make linear regression and gradient descent algorithm work correctly we need to make sure that features are on a similar scale.

![Feature Scaling](../../images/linear_regression/feature-scaling.svg)

For example "apartment size" feature (e.g. 120 m<sup>2</sup>) is much bigger than the "number of rooms" feature (e.g. 2).

In order to scale the features we need to do **mean normalization**

![Mean Normalization](../../images/linear_regression/mean-normalization.svg)

![x-i-j](../../images/linear_regression/x-i-j.svg) - _j<sup>th</sup>_ feature value of the _i<sup>th</sup>_ training example

![mu-j](../../images/linear_regression/mu-j.svg) - average value of _j<sup>th</sup>_ feature in training set

![s-j](../../images/linear_regression/s-j.svg) - the range (_max - min_) of _j<sup>th</sup>_ feature in training set.

## Polynomial Regression

Polynomial regression is a form of regression analysis in which the relationship between the independent variable _x_ and the dependent variable _y_ is modelled as an _n<sup>th</sup>_ degree polynomial in _x_.

Although polynomial regression fits a nonlinear model to the data, as a statistical estimation problem it is linear, in the sense that the hypothesis function is linear in the unknown parameters that are estimated from the data. For this reason, polynomial regression is considered to be a special case of multiple linear regression.

![Polynomial Regression](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Polyreg_scheffe.svg/650px-Polyreg_scheffe.svg.png)

Example of a cubic polynomial regression, which is a type of linear regression.

You may form polynomial regression by adding new polynomial features.

For example if the price of the apartment is in non-linear dependency of its size then you might add several new size-related features. 

![Polynomial Regression](../../images/linear_regression/polynomial-regression.svg)

## Normal Equation

There is a closed-form solution to linear regression exists and it looks like the following:

![Normal Equation](../../images/linear_regression/normal-equation.svg)

Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no “loop until convergence” like in gradient descent.

## Regularization

### Overfitting Problem

If we have too many features, the learned hypothesis may fit the **training** set very well:

![overfitting](../../images/linear_regression/overfitting-1.svg)

**But** it may fail to generalize to **new** examples (let's say predict prices on new example of detecting if new messages are spam).

![overfitting](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/t0zit.png)

### Solution to Overfitting

Here are couple of options that may be addressed:

- Reduce the number of features
    - Manually select which features to keep
    - Model selection algorithm
- Regularization
    - Keep all the features, but reduce magnitude/values of model parameters (thetas).
    - Works well when we have a lot of features, each of which contributes a bit to predicting _y_.

Regularization works by adding regularization parameter to the **cost function**:

![Cost Function](../../images/linear_regression/cost-function-with-regularization.svg)

> Note that you should not regularize the parameter ![theta zero](../../images/linear_regression/theta-0.svg).

![regularization parameter](../../images/linear_regression/lambda.svg) - regularization parameter

In this case the **gradient descent** formula will look like the following:

![Gradient Descent](../../images/linear_regression/gradient-descent-3.svg)

## References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [Linear Regression on Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent on Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [Gradient Descent by Suryansh S.](https://hackernoon.com/gradient-descent-aynk-7cbe95a778da)
- [Gradient Descent by Niklas Donges](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0)
- [Overfitting on GeeksForGeeks](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/)
