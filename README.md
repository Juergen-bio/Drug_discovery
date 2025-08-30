# Drug Discovery AI: Predicting Compound Activity (pIC50)

This project demonstrates a foundational machine learning pipeline for predicting the bioactivity of drug-like molecules. Using physicochemical descriptors and molecular fingerprints(Morgan), a regression model is trained to predict the `pIC50` values of compounds, a key metric for drug potency

-----

### Methodology

The project follows a standard cheminformatics machine learning workflow:

  * **Data Collection & Cleaning**: Data for potent molecules and their `pIC50` values were retrieved from the ChEMBL database. The data was cleaned to handle missing values and standardize the `pIC50` to a common scale.
  * **Feature Engineering**: Molecular structures, represented as SMILES strings, were converted into numerical features. Two types of features were used:
    1.  **Physicochemical Descriptors**: Properties like Molecular Weight (MW), LogP, and hydrogen bond donors/acceptors were calculated.
    2.  **Molecular Fingerprints**: Morgan fingerprints (`nBits=2048`, `radius=2`) were generated to encode the presence of specific molecular substructures.
  * **Feature Selection**: A `VarianceThreshold` filter was applied to remove low-variance fingerprint bits, reducing the feature space from over 2000 to just 60 (Pulling you own data depending on the date, can see this features change should the Chembl database be changed), and focusing the model on the most informative features.
  * **Model Training**: A **Random Forest Regressor** was trained on the prepared features. The model was trained on a subset of the data and evaluated on a separate, unseen test set to ensure robust performance.
  * **Model Evaluation**: The model's performance was evaluated using the **R-squared ($R^2$) score** and a visualization of predicted vs. actual values.

-----

### Results

The trained Random Forest model achieved a robust **$R^2$ score of 0.717** on the test set. This indicates the model can explain approximately 72% of the variance in the `pIC50` values, demonstrating strong predictive power. The plot of predicted vs. actual values visually confirms that the model's predictions align closely with the experimental data.

-----

### Repository Structure

  * `data_collection.ipynb`: A Jupyter notebook that collects the data from ChEMBL database.
  * `data_wrangling.ipynb`: Understanding the raw data and its parameters and cleaning it eg (removing null values).
  * `data_analysis.ipynb`: .Feature generation ie Morgan fingerpints
  * `fingerprinted_data.csv`: The cleaned and feature-engineered dataset containing all the final features for modeling.
  * `model_training.ipynb`: When the model learns and makes prediction.
  * `README.md`: This file.
  * `Requirement.yml`: Containes all the dependecies of this project (generated on the conda env).

-----

### Requirements

To run this project, you need the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn rdkit
```

-----

### Usage

1.  Clone this repository to your local machine.
2.  Install the required libraries.
3.  Open the `01_drug_discovery_ai.ipynb` Jupyter Notebook.
4.  Run all cells to execute the full pipeline from data cleaning to model evaluation.

-----

### Next Steps & Future Work

This project serves as a solid foundation. To expand on this work and advance towards a career in AI/ML for drug discovery, consider these steps:

  * **Advanced Modeling**: Implement and compare this model against more powerful techniques like **XGBoost** or **Graph Neural Networks (GNNs)**.
  * **Hyperparameter Tuning**: Optimize the Random Forest model's performance using techniques like `GridSearchCV` or `RandomizedSearchCV`.
  * **Alternative Prediction Tasks**: Apply this pipeline to predict other drug properties, such as toxicity or solubility.
  * **Larger Datasets**: Scale this project to use much larger datasets from sources like PubChem to build more generalizable models.
