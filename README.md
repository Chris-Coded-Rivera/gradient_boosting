# Car Sales Predictions Using Gradient Boosting Regression Models

In this project we built a regression model using gradient boosting to predict prices of a given vehicle. We used a dataset which can be found and dowloaded within this repo `car_sales.csv` for replication or your own personal use/experimentations.

This project was a great way to put into practice what I learned regarding gradient boosting. Let's get familiar with this algorithm, if you aren't already:

# Gradient Boosting

This algorithm is the result of combining 2 algorithms; gradient descent and boosting. Let's explain both, staring with __Gradient Descent__:

## Gradient Descent

To start, let's explain the 2 componenets that make up the core principles of gradient descent algorithm; __Loss Function Minimization__ & __Gradient Function__

- __Loss Function Minimization__;
    Once the task/objective for a task is identified (in this case we'll be predicting values of vehicles) a loss function must be chosen to evaluate a model's performance. In this project, we've chosen root mean squared error, or RMSE. The loss function is denoted as L(y,a), where <em>y</em> represents the true value and <em>a</a> represents the predicted value. The model's training task is to reduce loss after each iteration an thus the function representing such task is written belowe followed by RMSE loss function implementation:

Given 
y = true target value
a = predicted value
L = loss function
w = model weight

$$
w = arg \min_n L(y,a) \quad \Rightarrow \quad w = arg \min_n \sqrt {\sum{i=1}^n (y - a)^2}
$$

- __Gradient Function__:
    Gradient function is an efficient way models perform loss function minimization. The function is used to help determine the direction the values should follow to reach the loss function minimum. In machine learning, the values represent the model weights. The function is denoted as follows:

$$
\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},\frac{\partial f}{\partial x_3}\dots \frac{\partial f}{\partial x_n} \end{pmatrix}
$$

Where each partial derivatives indicates how quickly the functions change in direction of that variable, while keeping all other variables constant. 

In the case of our project's objective, we'll need to find the negative gradientas we need to find the direction the values of the model need to go to reduce RSME. The negative gradient is written as follows, where $x$ represents our parameter weights (assuming just 1 parameter for simplicity):

$$
- \nabla f(x)
$$

The gradient descent algorithm introduces the step size ($\mu$) variable which represents step size. The step size valiue indicates the magnitude the model will change in its $x$ value to find the loss function minimum. If the step size is too large, the model will miss the ideal weight to get the true loss function minimum. If too low, the model will take a long time to find the minimum. Gradient descent function is denoted in the following:

$$
x^1 = x^0 + \mu \times (-\nabla f(x))
$$

We will need to repeat over a set number of iterations in hopes to find the weight values that provide us with the minimum loss function. The formula is then written this way, where $t$ represents the iteration number:

$$
x^t = x^{t-1} + \mu \times (-\nabla f(x^{t-1}))
$$

Gradient descent is complete when the number of iterations is complete and/or the value of x no longer changes.

## Boosting

Boosting is an algorithm used commonly in __Ensemble__ models but tnot exclusively. ADABoost model is an example of a boosting model that is not use an ensemble. In this project we use 3 boosting ensemble models: CatBoost, XGBoost, and LightGBM. I wrote a desciption of each model with its pros and cons. We'll go over the algorithm of boosting while using an ensemble model.

Boosting using an ensemble model (Gradient Boosting Machines) essentially combine a bunch of models to improve the overall models performance by using the results of the previous models to improve the results of the next model by taking into account the errors of that previous model. It is a sequential of construction of an ensemble of models improving predictions at each step. 

Let's take a look at how it all works through the algorithm step by step:

$$
a_N (x) = \sum_{k=1}^N \gamma_k b_k (x)
$$

where aN(x) is the ensemble prediction. N represents the number of base learners, $\gamma_k$ is the model weight, and b_k is the base model prediction.

When we consider our regression task(as written below):

$$
RMSE(y,a) = \sqrt{\frac{1}{n} \sum_{i=1}^n (a(x_i)-y_i)^2} \longrightarrow \min_a(x)
$$

The model weights would be set to 1 to minimize the boosting function at the beginning of this section:

$\qquad \qquad \qquad \qquad \quad$$\gamma_k$ = 1, for all $k$=1,$\cdots n$

Which gets us this:

$$
a_N (x) = \sum_{k=1}^N b_k(x)
$$

With this, an ensemble of sequential models can be built using these formulas in sequence:

__First Step__:

Build base learner $b_1$ by solving the following minimization task:

$$
\begin{align*}
    b_1 = arg \min_b \sqrt{\frac{1}{n} \sum_{i=1}^1 (b(x_i)-y_i)^2}\\
    \downarrow \qquad \qquad \qquad \\ 
    a_1(x) = b_1(x) \qquad \qquad
\end{align*}
$$

The ensemble will then indicate the **residual**, or difference, between the prediction at the first step and the actual answer. The formula is as follows:

$$
e_{1,i} = y_i - b_1(x_i)
$$

__Second Step__:

The ensemble's mathematical process will then look like this:

$$
b_2 = arg \min_b \sqrt{\frac{1}{n} \sum_{i=1}^n (b(x_i)-e_{1,i})^2}
$$

The ensemble will take the following form and minimized to look like this:

$$
a_2(x) = \sum_{k=1}^2 b_k (x) = b_1(x) + b_2(x)
$$

We'll find the routine begin here when the residuals for each observation $i$ is calculated:

$$
e_2,i = y_i - a_2 (x_i) = y_i - \sum_{k=1}^2 b_k(x_i)
$$

At each step the algorithm will minimize the ensemble's error from the prior step.

# __Gradient Boosting__

When we combine all this we get gradient boosting. Let's revisit the ensemble formula:

$$
a_N(x) = a_{N-1}(x) + \gamma_N b_N(x)
$$

At each step the ensemble will select the answers that minimize the function, as demonstrated here:

$$
L(y,a(x)) \longrightarrow \min_a
$$

To minimize the function with gradient descent, at each step, the negative gradient of the loss function is calculated for $g_N$

$$
g_N(x) = -\nabla L(y,a_{N-1}(x) + a)
$$

To progre the predictions towards more accurate answers, the base learners learn to predict $g_N$ as demonstrated in the following formula:

$$
b_N(x) = arg \min_b \sqrt{\frac{1}{n}\sum_{i=1}^N (b(x_i)+g_N(x_i))^2}
$$

The weight for $b_N is obtained from the minimization task by iterating vaious numbers:

$$
\gamma_N = arg \min_\gamma L(y,a_{N-1}(x)+\gamma b_N(x))
$$

The coeddicient for the base learner helps adjust the ensemble make predictions as accurately as possible.

## In Summation

Gradient boosting is a viable algorithm to use for different loss functions that have derivatives such as (root) mean squared error or logarithmic in binary classification tasks.

I hope you enjoy the findings presented in this project's notebook!
