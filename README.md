# HDL Code Analysis and Generation Pipeline

A machine learning pipeline for analyzing and generating Hardware Description Language (HDL) code using transformer models and labeled data.

## Overview

This project provides tools for:
- Analyzing HDL code segments and patterns
- Training models on labeled HDL data
- Evaluating HDL code features
- Building towards HDL code generation

## Components

### HDL Analyzer
- Segments HDL code into meaningful components
- Labels code with features and purposes using LLM
- Stores analyzed data in MongoDB
- Reference: [labeler_with_llm.py, lines 12-165]

### Training Pipeline
- Loads labeled data from MongoDB
- Balances dataset for training
- Trains transformer models on code features
- Handles both segment and pattern classification
- Reference: [nlp.py, lines 33-286]

### Testing & Evaluation
- Predicts features of new HDL code
- Provides confidence scores for predictions
- Supports batch evaluation
- Reference: [test.py, lines 12-40]

## Setup

1. Install dependencies:
