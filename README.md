# YouTube Content Sentiment Analysis

## Overview

This project analyzes audience sentiment in YouTube comments across three different content domains:

- Travel
- Technology
- Cats

The objective is to explore how audience reactions, sentiment patterns, topics, and engagement differ across these domains.

## Initial Pilot Dataset

To validate the complete pipeline, the first version of the project will use:

- 3 YouTube creators
- 1 creator from each domain
- 5 videos per creator
- 50 comments per video
- 750 comments in total

The dataset will be expanded after the initial pipeline is working correctly.

## Project Workflow

1. YouTube data collection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Sentiment analysis
5. Topic and keyword analysis
6. Engagement analysis
7. Cross-domain comparison
8. Interactive dashboard

## Planned Sentiment Analysis

The project will initially investigate:

- VADER sentiment analysis
- Transformer-based sentiment analysis
- Comparison between sentiment approaches

## Project Structure

```text
youtube-content-sentiment-analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── config/
│
├── dashboard/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
