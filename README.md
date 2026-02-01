# LHON Variant Pathogenicity Classifier

##  Project Overview
This project applies Machine Learning to predict whether specific mitochondrial DNA (mtDNA) mutations associated with **Leber's Hereditary Optic Neuropathy (LHON)** are pathogenic or benign.

By engineering features based on amino acid physicochemical properties (Hydrophobicity, Charge, Molecular Weight), the model translates biological changes into numerical vectors for classification.

##  Key Features
* **Automated Data Processing:** Parses ClinVar data to extract protein-level mutations (e.g., `p.Arg340His`).
* **Bio-Physicochemical Feature Engineering:** Calculates the *Delta* (change) for:
    * Kyte-Doolittle Hydrophobicity
    * Molecular Weight (Steric hindrance)
    * Electric Charge (Arg/Lys vs Asp/Glu)
* **Model Optimization:** Progressed from a baseline Random Forest (40% accuracy) to a **Support Vector Machine (SVM) achieving 73.3% accuracy** using RBF kernels and Leave-One-Out Cross-Validation (LOOCV).

##  Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas` (Data Manipulation), `scikit-learn` (ML), `numpy` (Math)
* **Biology:** Mitochondrial genetics, Amino acid physicochemical properties

##  Results
| Model | Features Used | Accuracy |
| :--- | :--- | :--- |
| Random Forest (Baseline) | Hydrophobicity, MW | 40.0% |
| **SVM (Optimized)** | **Hydrophobicity, MW, Charge** | **73.3%** |

##  How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn numpy