# Sentiment-Analysis-for-Financial-News
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![Python 3.8](https://img.shields.io/badge/Python-3.8-brightgreen.svg) ![Spacy](https://img.shields.io/badge/Library-spaCy-orange.svg)

## What is Natural Language Processing?
Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written -- referred to as natural language. It is a component of artificial intelligence (AI).


## What is Sentiment Analysis?
Sentiment analysis refers to analyzing an opinion or feelings about something using data like text or images, regarding almost anything. 
Sentiment analysis helps companies in their decision-making process. For instance, if public sentiment towards a product is not so good, 
a company may try to modify the product or stop the production altogether in order to avoid any losses.

## Project Overview
In this project I am going to go through the classic Financial News Sentiment Analysis problem, which I will solve using the SpaCy library in Python.

## Problem definition
Given a bunch of financial news headlines the task is to predict whether a news headline contains positive, negative, or neutral sentiment. 
This is a typical supervised learning task where given a text string, we have to categorize the text string into predefined categories.

## Objectif
We're gonna build a ML model that is able to analyze a bunch of financial news headlines and be able to judge if 
they are of positive, negative or neutral sentiments.

## Solution
To solve this problem, we will follow the typical machine learning pipeline. We will first import the required libraries and the dataset. We will then do exploratory data analysis to see if we can find any trends in the dataset. Next, we will perform text preprocessing to convert textual data to numeric data that can be used by a machine learning algorithm. Finally, we will use machine learning algorithms to train and test our sentiment analysis models.

## Packages used
* SpaCy
* Pandas
* Sklearn
* Seaborn
* Matplotlib

## Dataset used
The dataset that we are going to use for this project is freely available at this [link](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news) on Kaggle

* Context:
This dataset contains the sentiments for financial news headlines from the perspective of a retail investor.

* Content:
The dataset contains two columns, "Sentiment" and "News Headline". The sentiment can be negative, neutral or positive.

## Implementation

### 1. Data Cleaning
We start by cleaning our data which means:
* Handling any missing data
* Tokenizing all the text 
* Lemmitization 
* Converting all characters to lower-case
* Removing stop words
* Remove punctuations

### 2. Data Preprocessing
In order to make text data understood to the machine learning algorithm we need a way to represent our text numerically. 
One tool can be used for doing this is called Bag of Words. BoW converts text into the matrix of occurrence of words within a given document. It focuses on whether given words occurred or not in the document, and it generates a matrix that we might see referred to as a BoW matrix or a document term matrix.

Here we're gonna use the `scikit-learn‘s TfidfVectorizer`. TF-IDF (Term Frequency-Inverse Document Frequency) is simply a way of normalizing our Bag of Words by looking at each word’s frequency in comparison to the document frequency. In other words, it’s a way of representing how important a particular term is in the context of a given document, based on how many times the term appears and how many other documents that same term appears in.

### 3. Training the models
In this step we are building and training a RandomForest and Adaboost Classifiers.

### 4. Make predictions and Evaluating the models
Once the model has been trained, the last step is to make predictions on the model trained to check if it will perform well on the unseen data by finally evaluating it using
the classification metrics such as: confusion metrix, F1 measure, accuracy, recall, presicion.
