# Neural-Network-for-Collision-Prediction
Purpose
The purpose of this project is to familiarize you with neural networks and their applications. More specifically, you will learn to collect training data for a robotics task and, in turn, design a neural network that can process this data. In the project your goal is to help a small robot navigate a simulated environment without any collisions. To accomplish this, you will need to train a neural network using backpropagation that predicts whether the robot is going to collide with any walls or objects in the next time step. All of the theories and best-practices introduced in the class are applicable here. Your task is to make sure the little fellow can safely explore its environment!
Objectives
Learners will be able to:
● Collect and manage a dataset used to train and test a neural network.
● Define and use PyTorch DataLoaders to manage PyTorch Datasets.
● Design your own neural network architecture in PyTorch.
● Evaluate and improve a neural network model and verify an application in simulation.
Technology Requirements
● System designed for use with Ubuntu 18.04
● Python and its related libraries. Using Anaconda is recommended
● Python libraries: cython matplotlib sklearn scipy pymunk pygame pillow numpy noise torch
Project Description
Part 1
The first task is to collect data that can be used to train the model. Please review “Image 1: PyGame Display” showing how the robot should wander around with no regard for its environment or avoiding collisions.
Collect a single sample per action containing, in order:
1. Thefive(5)distancesensorreadings
2. Theaction
3. Whetherornotacollisionoccurred(0:nocollision,1:collision)
Your data should be saved as a .csv file with seven (7) untitled columns. Please review “Image 2: Sample Learner Submission.CSV” of what your submission.csv should look like.
For grading purposes, submit your ‘submission.csv’ containing 100 data samples. For training in the future parts, you will need to collect much more than this.
Files to edit:
Collect_data.py
Image 1.0: Part 1 PyGame Display
Image 1.0: PyGame Display shows a sample environment and the robot’s potential actions. The robot should wander around with no regard for its environment or avoiding collisions. You may want to enlarge the image to see it clearly.
     2
 Image 2: Sample Learner Submission.CSV
Image 2: Sample Learner Submission.CSV shows what your submission.csv file should look like with seven (7) untitled columns. You may want to enlarge the image to see it clearly.
Part 2
Now that you have collected your training data, you can package it into an iterable PyTorch DataLoader for ease of use. You may be required to prune your collected data to balance out their distribution. If your dataset is 99% 0s and 1% 1s, a model that outputs only 0 would achieve good loss, but it would not have learned anything useful. Make sure to create both a training and testing DataLoader. Use training_data.csv collected from the previous part. Make sure to use the PyTorch classes mentioned in the comments of Data_Loaders.py.
Files to edit:
Data_Loaders.py saved/training_data.csv
     3

Part 3
For Part 3, you will be designing your own custom neural network using PyTorch’s torch.nn class. You will need to initialize a custom architecture, define a forward pass through the network, and build a method for evaluating the fit of a given model.
Files to edit:
Data_Loaders.py Networks.py saved/*
Part 4
In Part 4, you must train a model using your custom network architecture, which accurately predicts collisions given sensor and action information. Your grade will depend on the accuracy of the model. You may need to try many different strategies and architectures to achieve a well fit model. Keep track of your training and testing loss throughout the epochs, and generate a plot with these lines at the end. To see an application demo of the learning your robot has done, run goal_seeking.py, which will have the robot seek out goals while only taking possible actions it deems to be safe. Please review “Image 1.1: Part 4 PyGame Display.”
Files to edit:
Data_Loaders.py Networks.py train_model.py saved/*
Image 1.1: Part 4 PyGame Display
Image 1.1: Part 4 PyGame Display shows the plot of the training and testing loss lines. You may want to enlarge the image to see it clearly.
 
