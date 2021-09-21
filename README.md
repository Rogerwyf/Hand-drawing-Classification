# Hand-drawing-Classification

Final Report ([PDF Version](https://github.com/Rogerwyf/Hand-drawing-Classification/blob/main/report/final_report.pdf))

## Background
[Quick, Draw!](https://quickdraw.withgoogle.com/) is an online game developed by
Google in which users are given one of 345 categories,
and prompted to create an on-screen drawing corre-
sponding to that category. As the user draws, the sys-
tem has up to 20 seconds to guess the category. Over
50 million drawings of users have been collected and
made available as an open-sourced dataset of 28x28
bitmaps.

## Project Objective
This project serves as the final project for STAT 441 - Classification at the University of Waterloo. 
The purpose of this project is to answer the question on the Quick, Draw! homepage: "Can a neural 
network learn to recognize doodling?" To be specific, we look at the performance of different models 
on structurally similar and different doodling classification problem.

Models that we experimented with

    - CNN
    - FFNN
    - ResNet
    - Xgboost

We finally experimented with **Word Embeddings** and **Transfer Learning** with different models on this problem.

## Environment Setup
Use `env.yml` to replicate the environment.
```
conda create --name HDC --file env.yml
```

## Code Overview
For PyTorch based model creation, training and analysis, see `*.py` python files under `src` directory. 

For TensorFlow based model creation, training and analysis, see `*.ipynb` jupyter notebook files under `notebooks` 
directory.


