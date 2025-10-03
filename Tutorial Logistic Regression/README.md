# Logistic Regression Tutorial

A comprehensive tutorial on logistic regression and its applications in decision-making models, specifically focusing on evidence accumulation models in neuroscience and brain-computer interfaces (BCI).

## 📚 Contents

### Main Tutorials

#### 1. **Tutorial_Logistic_Regression.ipynb**
**Core logistic regression concepts and implementation**

**Learning Objectives**:
- Understand the mathematical foundations of logistic regression
- Implement logistic regression from scratch
- Apply logistic regression to classification problems
- Interpret model coefficients and predictions
- Evaluate model performance using appropriate metrics

**Key Topics**:
- Sigmoid function and logit transformation
- Maximum likelihood estimation
- Gradient descent optimization
- Regularization techniques (L1/L2)
- Model evaluation and validation

#### 2. **Deep_Learning_BCI_Tutorial.ipynb**
**Brain-computer interface applications using deep learning**

**Learning Objectives**:
- Understand BCI signal processing and classification
- Implement neural networks for EEG signal analysis
- Apply deep learning to motor imagery classification
- Evaluate BCI system performance
- Understand real-time BCI applications

**Key Topics**:
- EEG signal preprocessing and feature extraction
- Convolutional neural networks for time-series data
- Motor imagery classification
- Real-time signal processing
- BCI performance metrics and evaluation

### Data Generation

#### **Data_Generator.ipynb**
**Synthetic data generation for tutorial exercises**

**Purpose**: Generate controlled datasets for practicing logistic regression and evidence accumulation models

**Features**:
- Configurable data parameters
- Multiple decision-making scenarios
- Evidence accumulation simulation
- Noise and variability controls
- Export capabilities for analysis

## 🧠 Evidence Accumulation Models

### Theoretical Background

Evidence accumulation models are computational frameworks that describe how decisions are made through the gradual accumulation of evidence over time. These models are particularly relevant in:

- **Neuroscience**: Understanding neural decision-making processes
- **Psychology**: Modeling choice behavior and reaction times
- **BCI Applications**: Real-time decision support systems
- **Robotics**: Autonomous decision-making algorithms

### Key Concepts

#### 1. **Drift Diffusion Model (DDM)**
- Continuous evidence accumulation
- Decision boundaries and thresholds
- Reaction time distributions
- Choice probability calculations

#### 2. **Race Models**
- Competing evidence accumulators
- First-passage time distributions
- Multi-alternative decision making
- Parallel processing architectures

#### 3. **Logistic Regression Integration**
- Evidence strength quantification
- Decision boundary estimation
- Confidence assessment
- Uncertainty quantification

## 🗂️ Dataset Structure

### Data Organization
```
data/
├── 0/          # Class 0 samples
├── 1/          # Class 1 samples  
├── 2/          # Class 2 samples
├── 3/          # Class 3 samples
└── 4/          # Class 4 samples
```

### Data Characteristics
- **Multi-class classification**: 5 distinct classes
- **Evidence accumulation**: Time-series decision data
- **Synthetic generation**: Controlled experimental conditions
- **Real-world simulation**: Realistic noise and variability

## 🎨 Visualizations

### Experiment Visualization
- **`Experiment.png`**: Overview of experimental setup
- **`Experiment_Explained.png`**: Detailed explanation of methodology
- **`logisticRegressionCartoon.png`**: Conceptual illustration of logistic regression
- **`Results_Evidence_Accumulation.png`**: Evidence accumulation results

### Key Plot Types
- **Decision boundaries**: 2D and 3D visualization of classification regions
- **Evidence accumulation curves**: Time-series plots showing evidence build-up
- **ROC curves**: Performance evaluation across different thresholds
- **Confusion matrices**: Detailed classification performance analysis
- **Feature importance**: Coefficient analysis and interpretation

## 🛠️ Technical Implementation

### Required Packages
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
```

### Key Algorithms

#### 1. **Logistic Regression Implementation**
```python
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Implementation of gradient descent
        pass
    
    def predict(self, X):
        # Prediction using learned parameters
        pass
```

#### 2. **Evidence Accumulation Model**
```python
class EvidenceAccumulator:
    def __init__(self, drift_rate, threshold, noise):
        self.drift_rate = drift_rate
        self.threshold = threshold
        self.noise = noise
    
    def accumulate_evidence(self, evidence, time_steps):
        # Simulate evidence accumulation process
        pass
    
    def make_decision(self, accumulated_evidence):
        # Decision based on accumulated evidence
        pass
```

## 🎯 Learning Objectives

After completing this tutorial, you will be able to:

1. **Understand Logistic Regression**: Master the mathematical foundations and implementation
2. **Apply to Decision Making**: Use logistic regression in evidence accumulation contexts
3. **Implement BCI Systems**: Build brain-computer interface classification systems
4. **Evaluate Performance**: Assess model performance using appropriate metrics
5. **Interpret Results**: Understand and communicate model outputs effectively

## 🔬 Real-World Applications

### Neuroscience Research
- **Decision-making studies**: Understanding how brains make choices
- **Neural signal analysis**: Processing EEG, MEG, and other neural data
- **Cognitive modeling**: Simulating human decision processes
- **Clinical applications**: Diagnostic and therapeutic decision support

### Brain-Computer Interfaces
- **Motor imagery classification**: Controlling devices with brain signals
- **Communication systems**: Alternative communication for disabled individuals
- **Neuroprosthetics**: Controlling artificial limbs and devices
- **Gaming and entertainment**: Brain-controlled games and applications

### Machine Learning Applications
- **Binary and multi-class classification**: General-purpose classification tasks
- **Feature selection**: Identifying important predictors
- **Model interpretation**: Understanding decision-making processes
- **Ensemble methods**: Combining multiple logistic regression models

## 📊 Performance Evaluation

### Classification Metrics
- **Accuracy**: Overall classification performance
- **Precision and Recall**: Per-class performance analysis
- **F1-Score**: Balanced performance metric
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed error analysis

### Evidence Accumulation Metrics
- **Decision Time**: Time to reach decision threshold
- **Choice Probability**: Probability of correct decisions
- **Confidence Calibration**: Alignment between confidence and accuracy
- **Evidence Strength**: Quantification of decision evidence

## 🎓 Tutorial Structure

### Beginner Level
1. **Basic Logistic Regression**: Understanding the fundamentals
2. **Simple Classification**: Binary classification examples
3. **Model Interpretation**: Understanding coefficients and predictions

### Intermediate Level
1. **Multi-class Classification**: Extending to multiple classes
2. **Feature Engineering**: Improving model performance
3. **Regularization**: Preventing overfitting

### Advanced Level
1. **Evidence Accumulation Models**: Complex decision-making scenarios
2. **BCI Applications**: Real-time signal processing
3. **Deep Learning Integration**: Neural network approaches

## 🤝 Contributing

We welcome contributions to improve this tutorial:
- **Additional Examples**: New use cases and applications
- **Enhanced Visualizations**: Better plots and interactive elements
- **Code Improvements**: Optimization and bug fixes
- **Documentation**: Clearer explanations and examples
- **New Datasets**: Additional data for practice

## 📚 Additional Resources

### Research Papers
- "The Drift Diffusion Model: A Theory of Decision Making" by Ratcliff and McKoon
- "Logistic Regression: A Self-Learning Text" by Kleinbaum and Klein
- "Brain-Computer Interfaces: Principles and Practice" by Wolpaw and Wolpaw

### Online Resources
- Scikit-learn Logistic Regression documentation
- TensorFlow tutorials for deep learning
- EEG analysis tutorials and toolboxes
- BCI research community resources

### Tools and Software
- **MNE-Python**: EEG/MEG data analysis
- **BCILAB**: MATLAB toolbox for BCI research
- **OpenBCI**: Open-source BCI hardware and software
- **PsychoPy**: Psychology experiment design

---

*This tutorial provides a comprehensive introduction to logistic regression with practical applications in neuroscience, decision-making, and brain-computer interfaces, bridging theoretical concepts with real-world implementations.*
