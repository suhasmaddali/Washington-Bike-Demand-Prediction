 # 🚴‍♀️ Washington Bike Demand Prediction

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 

There are different bike companies such as [__Zagster__](https://en.wikipedia.org/wiki/Zagster) and [__Lime Bikes__](https://www.li.me/en-US/home) present all over the __US__ to ensure that people get rides when needed. As a result, there is an increase in demand for these bikes as some people move to these __cost-efficient__ and __environment-friendly__ transport options. As a result of this transition, there is variation in the demand for bikes in different regions. 

<img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Bike%20Demand%20Prediction%20Image.jpg" width = 1000 height = "600"/>

## Challenges

One of the challenges that these companies face is to know the __number of bikes__ that must be placed at different locations at different instances of time to ensure that they get __maximum profit__ and give the people their rides. Sometimes there is a possibility of people missing out these bikes due to their unavailability. On the contrary, there are also instances when the demand for these bikes are low while they are highly available in many locations without being used. Therefore, it becomes important to tackle these instances and understand the demand for the bikes for different days and scenarios. 

## Data Science and Machine Learning 

With the help of __machine learning__ and __deep learning__, this problem could be addressed and this would ensure that the demand for the bikes are known beforehand and thus, the companies could ensure that there are __adequate bikes__ present in different locations.

## Exploratory Data Analysis (EDA)

Therefore, with the help of __data visualization__ and __machine learning__, bike rental companies would be able to understand the total number of bikes that must be present at different instances of time and thus, they would be able to predict the demand for the bikes in the future. This would ensure that the companies save millions of dollars by giving the right service to different people who are in need. 

## Metrics

Since this is a __regression__ problem, we consider metrics that take into account __continuous output variables__ and give their estimates based on the difference between the __actual output__ and __predicted output__. Below are some metrics that are used for this prediction.

* [__Mean Squared Error__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Machine Learning Models

There are a large number of machine learning models used in the prediction of the demand for __Washington Bikes__. Below are the models that were used for prediction.

* [__Deep Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
* [__K Nearest Neighbors__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* [__Partial Least Squares (PLS) Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
* [__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [__Long Short Term Memory (LSTM)__](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

## Machine Learning Predictions and Analysis 

It is important to know some of the features that are present in the data so that we could be doing the __machine learning__ analysis. We would be performing various __data visualizations__ to understand some of the underlying features and once we get a good understanding of them, we are going to be using different machine learning models for predicting the demand for bikes based on these features. Once we get the machine learning predictions, we are going to be using different strategies that could aid us in the process of productionizing the models that could be used in different ways in companies. Therefore, this would save a lot of time and money for the bike rental companies where the demand for the bikes is predicted beforehand with the aid of machine learning and deep learning respectively. 

## Outcomes
* The models that were able to perform the best for predicting the bike demand are __Gradient Boosting Decision Regressor__ and __Deep Neural Networks__.
* __Exploratory Data Analysis (EDA)__ was performed to ensure that there is a good understanding of different features and their contribution to the output variable respectively. 
* The best machine learning models were able to generate a __mean absolute error (MAE)__ of about __23__ which is really good considering the scale of the problem at hand.

## Future Scope
* Additional features such as the __street connectivity score__ and __people's perceptions__ about the bicycling environment could be added to generate even more good predictions from the models. 
* The best machine learning models could be __deployed__ in real-time where the demand for bikes is highlighted so that admins can take action based on the __demand__. 

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git that could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "hhttps://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link of the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(18).png" width = "600" />

5. The link of the repository can be found when you click on "Code" (Green button) and then, there would be a html link just below. Therefore, the command to download a particular repository should be "Git clone html" where the html is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(20).png" width = "600" />

8. Later, open the jupyter notebook by writing "jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Washington-Bike-Demand-Prediction/blob/main/images/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 





































