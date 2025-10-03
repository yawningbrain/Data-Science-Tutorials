# Introduction to Statistics

A comprehensive collection of 9 tutorials covering fundamental statistical concepts from basic descriptive statistics to advanced hypothesis testing. Each tutorial builds upon previous knowledge and includes hands-on exercises with real datasets.

## 📚 Tutorial Overview

### Tutorial 1: Finding Outliers using the Empirical and IQR Rules
**File**: `Tutorial 1 Finding Oultiers using the Emperical and IQR Rules.ipynb`

**Learning Objectives**:
- Understand different methods for outlier detection
- Apply the Empirical Rule (68-95-99.7 rule) for normal distributions
- Use the Interquartile Range (IQR) method for robust outlier detection
- Visualize outliers using box plots and scatter plots

**Key Concepts**:
- Standard deviations and z-scores
- Quartiles and percentiles
- Box plot interpretation
- Robust vs. non-robust statistical measures

**Datasets Used**: Various datasets for outlier analysis

### Tutorial 2: Binomial Distribution
**File**: `Tutorial 2 Binomial Distribution.ipynb`

**Learning Objectives**:
- Understand the properties of binomial distributions
- Calculate probabilities using the binomial formula
- Apply the normal approximation to binomial distributions
- Solve real-world problems involving binary outcomes

**Key Concepts**:
- Bernoulli trials and binomial experiments
- Probability mass function
- Mean and variance of binomial distributions
- Continuity correction
- Applications in quality control and surveys

### Tutorial 3: Gaussian Distribution
**File**: `Tutorial 3 Gaussian Distribution.ipynb`

**Learning Objectives**:
- Master the properties of normal (Gaussian) distributions
- Calculate probabilities using standard normal tables
- Apply the Central Limit Theorem
- Perform z-score transformations

**Key Concepts**:
- Normal distribution properties
- Standard normal distribution (Z-distribution)
- Probability density function
- Cumulative distribution function
- Applications in natural phenomena and measurement errors

### Tutorial 4: Central Limit Theorem
**File**: `Tutorial 4 Central Limit Theorem.ipynb`

**Learning Objectives**:
- Understand the Central Limit Theorem and its implications
- Demonstrate CLT through simulation
- Apply CLT to sample means and proportions
- Understand the role of sample size

**Key Concepts**:
- Sampling distributions
- Law of large numbers
- Convergence in distribution
- Practical applications in statistical inference
- Simulation-based learning

### Tutorial 5: Hypothesis Testing
**File**: `Tutorial 5 Hypothesis Testing.ipynb`

**Learning Objectives**:
- Formulate null and alternative hypotheses
- Understand Type I and Type II errors
- Calculate p-values and make statistical decisions
- Interpret results in context

**Key Concepts**:
- Hypothesis testing framework
- Significance levels and critical values
- One-tailed vs. two-tailed tests
- Power of statistical tests
- Effect size and practical significance

### Tutorial 6: Review Session with Colab Integration
**File**: `Tutorial 6 Review Session with Colab Integration.ipynb`

**Learning Objectives**:
- Review key concepts from previous tutorials
- Practice with Google Colab environment
- Integrate multiple statistical concepts
- Prepare for advanced topics

**Key Concepts**:
- Comprehensive review of statistical fundamentals
- Cloud-based computing with Colab
- Collaborative data analysis
- Best practices for statistical analysis

### Tutorial 7: t-Tests Statistics
**File**: `Tutorial 7 t Tests Statistics.ipynb`

**Learning Objectives**:
- Understand when to use t-tests vs. z-tests
- Perform one-sample, two-sample, and paired t-tests
- Calculate confidence intervals for means
- Interpret t-test results

**Key Concepts**:
- t-distribution properties
- Degrees of freedom
- Independent vs. dependent samples
- Assumptions of t-tests
- Effect size measures (Cohen's d)

### Tutorial 8: A/B Testing
**File**: `Tutorial 8 AB Testing.ipynb`

**Learning Objectives**:
- Design and analyze A/B tests
- Calculate statistical power and sample size
- Handle multiple testing corrections
- Interpret A/B test results for business decisions

**Key Concepts**:
- Experimental design principles
- Randomization and control groups
- Statistical significance vs. practical significance
- Sequential testing and early stopping
- Real-world A/B testing scenarios

### Tutorial 9: R-Squared and Data Fitting
**File**: `Tutorial 9 R Squared and Data Fitting.ipynb`

**Learning Objectives**:
- Understand R-squared and adjusted R-squared
- Perform linear regression analysis
- Evaluate model fit and assumptions
- Compare different regression models

**Key Concepts**:
- Coefficient of determination (R²)
- Residual analysis
- Model assumptions and diagnostics
- Overfitting and model selection
- Multiple regression basics

## 🗂️ Datasets

The `Datasets/` folder contains curated datasets for hands-on practice:

### Available Datasets:
- **`beer_small.csv`** & **`beers.csv`**: Beer-related data for analysis
  - Use cases: Descriptive statistics, outlier detection, correlation analysis
- **`co-emissions-per-capita.csv`**: Worldwide CO2 emissions per capita data
  - Use cases: Time series analysis, international comparisons, trend analysis
- **`dietary-compositions-by-commodity-group.csv`**: Nutritional dataset
  - Use cases: Categorical data analysis, correlation studies, dietary patterns
- **`WA_Fn-UseC_-HR-Employee-Attrition.csv`**: HR employee attrition data
  - Use cases: Logistic regression, hypothesis testing, employee analytics
- **`linear regression_raw_data`**: Linear regression practice data
  - Use cases: Regression analysis, model fitting, residual analysis

## 🎯 Learning Path

### Recommended Sequence:
1. **Tutorials 1-3**: Foundation (Outliers, Binomial, Gaussian)
2. **Tutorial 4**: Central Limit Theorem (bridges descriptive to inferential)
3. **Tutorial 5**: Hypothesis Testing (core inferential statistics)
4. **Tutorial 6**: Review and Integration
5. **Tutorials 7-9**: Advanced Applications (t-tests, A/B testing, regression)

### Prerequisites by Tutorial:
- **Tutorials 1-3**: Basic algebra and probability
- **Tutorial 4**: Understanding of sampling
- **Tutorial 5**: Completion of Tutorials 1-4
- **Tutorial 6**: All previous tutorials
- **Tutorial 7**: Understanding of hypothesis testing
- **Tutorial 8**: Statistical inference concepts
- **Tutorial 9**: Basic understanding of correlation

## 🛠️ Technical Requirements

### Required Packages:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
```

### Installation:
```bash
pip install numpy pandas matplotlib seaborn scipy plotly
```

## 📊 Key Statistical Concepts Covered

### Descriptive Statistics:
- Measures of central tendency (mean, median, mode)
- Measures of variability (range, variance, standard deviation)
- Distribution shapes and skewness
- Outlier detection methods

### Probability Distributions:
- Binomial distribution and applications
- Normal (Gaussian) distribution
- t-distribution
- Sampling distributions

### Inferential Statistics:
- Central Limit Theorem
- Hypothesis testing framework
- Confidence intervals
- Type I and Type II errors
- Statistical power

### Advanced Topics:
- t-tests (one-sample, two-sample, paired)
- A/B testing and experimental design
- Linear regression and model evaluation
- Effect size and practical significance

## 🎨 Visualization Techniques

Each tutorial includes comprehensive visualizations:
- **Histograms and density plots** for distribution analysis
- **Box plots and violin plots** for outlier detection
- **Scatter plots** for correlation analysis
- **Q-Q plots** for normality assessment
- **Residual plots** for regression diagnostics
- **Interactive plots** using Plotly for exploration

## 🔬 Real-World Applications

### Business Applications:
- Quality control and process improvement
- Marketing A/B testing
- Employee analytics and HR decisions
- Financial risk assessment

### Scientific Applications:
- Experimental design and analysis
- Survey research and polling
- Medical and pharmaceutical research
- Environmental data analysis

### Data Science Applications:
- Model validation and evaluation
- Feature selection and engineering
- Statistical significance testing
- Experimental design for ML

## 🎓 Assessment and Practice

### Each Tutorial Includes:
- **Conceptual questions** to test understanding
- **Computational exercises** for hands-on practice
- **Real-world problem solving** with actual datasets
- **Extension activities** for advanced learners

### Recommended Practice:
1. Complete all code examples
2. Work through exercises independently
3. Experiment with different datasets
4. Apply concepts to your own data
5. Discuss results and interpretations

## 🤝 Contributing

We welcome contributions to improve these tutorials:
- **Error corrections** and bug fixes
- **Additional examples** and use cases
- **Enhanced visualizations** and interactivity
- **New datasets** for practice
- **Improved explanations** and clarity

## 📚 Additional Resources

### Recommended Reading:
- "Introduction to Statistical Thought" by Michael Lavine
- "The Practice of Statistics" by Daren Starnes
- "Statistical Rethinking" by Richard McElreath
- "Think Stats" by Allen Downey

### Online Resources:
- Khan Academy Statistics courses
- Coursera Statistical Learning courses
- MIT OpenCourseWare Statistics
- R-bloggers for practical applications

---

*This collection provides a solid foundation in statistical thinking and analysis, preparing students for advanced topics in data science, machine learning, and scientific research.*
