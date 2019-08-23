# MachineLearningForWaterGlycerolViscosity
We used different machine learning algorithms to predict the viscosity of a water-glycerol mixture as a function of the concentration of the glycerol and of the temperature of the mixture starting from an existing database. In the database the viscosity (cP) of the mixture is given as a function of the glycerol concentration (*Wt %*) and of the temperature (*°C*). These are actual experimental data taken from J.B. Segur and H. Oberstar, *"Viscosity of glycerol and its acqueous solutions"*, Ind. Eng. Chem.19514392117-2120.

The machine learning algorithms we will consider are:
* Linear regression
* K-Nearest Neighbours
* Decision tree and random forest
* Support Vector Machine
* Artificial Neural Network

The performance of the Machine Learning algorithms will also be compared with a state-of-the-art correlation taken from Cheng, N.S., Formula for the viscosity of a water-glycerol mixture, *Industrial & Engineering Chemistry Research*, 47, 3285-3288 (2008)

All the Machine Learning analysis will be performed using the well-known [Scikit Learn](https://scikit-learn.org/stable/) Python library within the [Jupyter Notebook](https://jupyter.org/) environment. A home-made Python module will be used to perform an ANOVA analysis for the linear regression.

More information can be found in [this blogpost](https://www.centds.com/2019/08/23/predict-the-viscosity-of-a-water-glycerol-mixture-using-machine-learning/).

