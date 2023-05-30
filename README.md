
# PROGRAMMING MODEL OF EXPLOITATION TIMES FOR DECISION MAKING IN THE DISTRIBUTION OPPORTUNITIES OF COLOMBIAN FILM PROJECTS.

## Sumary

The film industry differs from others by its unpredictable scenarios, which are difficult to model
quantitatively and require the intuition of subject matter experts. However, in recent years, the non-analytical has been neutralized by a growing body of academics and pioneers, who have successfully demonstrated that, despite its dynamic complexities, the decision-making environment in the film can be modeled mathematically. 
The research project is an approach to modeling the Colombian industry that ranges from the analysis of the distribution of Colombian films to the application of two Multinomial Logistic Regression (MLR) models that try to predict from a set of variables the number of weeks and release date of a film in Colombia.

## Thesis Development

1. The distribution windows for feature films in Colombia were identified, namely: Theatrical, VOD, DVD, public television, and private television.
2. A repository was created with Colombian films released between 2010 and 2019, aiming to capture all available information.
3. The data available in the repository was explored, and pathway for the research paper development were proposed.
4. Distribution windows other than Theatrical were discarded due to insufficient data and lack of homogeneity among the available windows.
5. Preprocessing of the available Theatrical data was performed, and some records were estimated to retain the maximum possible amount of information.
6. The Theatrical data was re-explored with the objective of identifying relationship patterns.
7. Various mathematical models applied in the industry were analyzed, including: Linear regression, logistic regression, neural networks, decision trees, discriminant analysis and genetic algorithms.
7. A model was identified from the literature that demonstrated a good prediction success rate in the industry and the possibility to learn by our own.
8. Multinomial Logistic Regression was selected.
9. Autonomous learning of the concepts of the Multinomial Logistic Regression model was undertaken.
10. Autonomous learning of algorithm design and application in MATLAB was undertaken.
11. The model was applied to the available data.
12. Factors influencing the model's efficiency were analyzed.
13. Finally, the project was concluded.

## Multinomial Logistic Regression Applied to Predicting the Release Date of Colombian Films

The data used in the file `Dataset_releasedate.cvs` for this model was collected between 2020 and 2021 during the development of my thesis project or research paper. The data comes from different sources, including Proimagenes, reports from the Colombian government, IMDB Pro, etc. Please note that I cannot guarantee the veracity of the data. The dataset consists of 260 records with 10 columns (attributes). 

**Note**

The original code programming in MATLAB was not available, This recreation was made based on the available information, the original Dataset. More detailed documentation of the project can be found in the PDF file in this folder, but please note that it is in Spanish.

## Objetive

The objective of this application was to evaluate the effectiveness of multinomial logistic regression in predicting the dependent variable "release date" in the context of movie premieres. The "release date" variable consisted of 12 possibilities corresponding to the months of the year. In the original thesis, two applications of this technique were conducted, one for the release date and the other one for the possible duration in the Theatrical window. However, for the purpose of presenting an English summary for my application for the master's degree, the code was implemented in Python. The research received a rating of 46/50, which earned it a merit distinction.

### Installation

1. Clone the repository or download the code files.
2. Ensure you have Python (version 3.x or higher) installed.
3. Install the required dependencies by running the following command:
   ```shell
   pip install pandas 
   pip install numpy 
   pip install scikit-learn
   ```

### Usage

1. Place the dataset file (`Dataset_releasedate.csv`) in the same directory as the code files.
2. Open the code file (`logistic_regression_a.py`) in your preferred Python IDE or text editor.
3. Modify the code file if necessary, such as changing hyperparameters or file paths.
4. Run the code to train the logistic regression model and predict the release dates of Colombian films.
5. The predicted release dates will be evaluated and the accuracy will be displayed.

### Conclusions

- The logistic regression applied to predict the launch date yielded an accuracy of 15.38% (0.1538). The dataset was distributed as follows: 95% for training and 5% for testing. This distribution was done due to the availability of only 260 records.

- When mathematically modeling a problem, it is important to consider various factors. Two vital factors include the quantity and quality of the data in the dataset. Both factors complement each other, and emphasizing one over the other can lead to issues. For a classification model, it would be ideal to have at least 1000 records, and even then, there is a risk of overfitting. Regarding data quality, it is necessary for the data to be significant for the problem, and its integrity should be validated by an entity.

