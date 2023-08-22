# A Holistic AI-based Approach for Pharmacovigilance Optimization from Patients Behavior Social Media

## Description

This repository contains the code corresponding to the paper ``**A Holistic AI-based Approach for Pharmacovigilance Optimization from Patients Behavior Social Media**'' submitted to the Artificial Intelligence in Medicine journal. https://www.sciencedirect.com/science/article/abs/pii/S0933365723001525

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Installation
1. Clone the repository:
```git clone https://github.com/SMART-Lab-NYU/EarlyDetection_PharmaceuticalScandals.git```

2. Navigate to the project directory:
```cd [project directory]```

3. Install the dependencies:
This code was tested with Python 3.
```pip install pandas```
```pip install numpy```
```pip install matplotlib```
```pip install nltk```
```pip install spacy```
```python -m spacy download fr_core_news_sm```


# Usage

1. Scraping (skip this step for reproductible results from the paper)

- **1_scraping.py**: Scrap data from the Doctossimo Forum.
Please note that this scraping algorithm is currently dysfunctional due to a change in the html structure of the doctissimo.fr site.
Now class names in div tags are randomly generated on the fly; which makes it difficult to retrieve the data via our algorithm (or the Beautifulsoup python library).
This website is certainly not the best source of medical information, moreover, it was a demonstrator to carry out this work. We have therefore made the decision not to maintain this algorithm.
For the second step you can scrap your own data or use the file named "dataset_doctissimo_22_03_2020.csv" in data's files.

2. Starting point for reproducible results
- **2_datasets_formating_cleaning.py**: Perform cleaning and formatting steps from "dataset_doctissimo_22_03_2020.csv" file.
- **3_word_frequency_analysis.py**: Perform word frequency analysis.
- **4_fasttext_word2vec_mlmodel**: Train model for generating word embeddings. 
- **5_sentiment_analysis_CNN_clustering**: Apply sentiment analysis. 
- **6_side_effects_distribution_correlation**: Correlation analysis for side effects. 
- **7_convolution_neural_network**: Train deep learning models. 

# Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Commit your changes and push the branch to your forked repository.
5. Submit a pull request detailing your changes.
6. Please ensure that your contributions adhere to the project's coding conventions and follow best practices. Also, make sure to include appropriate documentation and tests for any new features.

# License

The project is distributed under the Apache License 2.0. 

# Contact

If you have any questions, suggestions, or feedback, please feel free to reach out to hanan.salam(at)nyu(dot)edu.

# Acknowledgments

If you use this code, please cite the folllowing paper: 

@article{roche2022ai,
  title={A Holistic AI-based Approach for Pharmacovigilance Optimization from Patients Behavior Social Media},
  author={Roche, Valentin and Robert, Jean-Philippe and Salam, Hanan},
  journal={arXiv preprint arXiv:2203.03538},
  year={2022}
}





