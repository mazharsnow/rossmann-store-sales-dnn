# Rossmann Store Sales – Deep Neural Network

This repository contains a Deep Neural Network (DNN) solution for the  
[Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales).

The goal is to forecast daily sales for 1,115 Rossmann stores across Germany using
historical sales, store metadata, competition, promotions, and calendar information.

---

## 1. Project Overview

**Objective**

- Build a Deep Neural Network using **TensorFlow/Keras**.
- Train it on the official Rossmann dataset.
- Generate a valid `submission.csv` for Kaggle.

**Main ideas**

- Model the target as **`log1p(Sales)`** to stabilize variance.
- Use a rich set of **engineered features**:
  - Calendar: `Year`, `Month`, `Day`, `WeekOfYear`, `DayOfWeek`
  - Competition: `CompetitionDistance`, `HasCompetition`, `CompetitionOpenMonths`
  - Promotions: `Promo`, `Promo2`, `Promo2OpenWeeks`, `IsPromoMonth`
  - Store aggregates: `StoreAvgSales`, `StoreMedianSales`, `StoreDowAvgSales`
  - Lag & rolling: `Sales_lag_1`, `Sales_lag_7`, `Sales_lag_14`,
    `Sales_rollmean_7`, `Sales_rollmean_30`

**Performance (time-based validation)**

- Baseline DNN (without aggregates/lag features): ~**0.259** RMSPE  
- Final DNN (with store aggregates + lag/rolling features):  
  - **Best** validation RMSPE ≈ **0.118**  
  - Typical runs around **0.15**, due to training randomness  

The script produces a `submission.csv` in the correct format to submit to Kaggle.

---

**## 3. Environment Setup**

### 3.1 Create & activate a virtual environment

From the project root (where `rossmann_dnn.py` lives):

**Windows (PowerShell)**
```powershell
# Create
python -m venv venv
# Activate
.\venv\Scripts\Activate.ps1

3\. Environment Setup
---------------------

### 3.1 Create and activate a virtual environment

From the project root (where rossmann\_dnn.py lives):

**Windows (PowerShell)**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Create virtual environment  python -m venv venv  # Activate  .\venv\Scripts\Activate.ps1   `

**Linux / macOS**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m venv venv  source venv/bin/activate   `

After activation, your prompt should show something like (venv).

### 3.2 Install dependencies

With the environment activated:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

(Optional) verify TensorFlow is installed and using the venv’s Python:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -c "import tensorflow as tf, sys; print(tf.__version__, sys.executable)"   `

If you want to disable oneDNN custom ops warnings in **PowerShell**:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   $env:TF_ENABLE_ONEDNN_OPTS="0"   `

4\. Data
--------

The project expects the following CSV files to be present in the data/ folder:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data/  ├── train.csv  ├── test.csv  └── store.csv   `

In this setup, these files are **already placed** in the data/ directory, so no extra download step is required to run the code.

> If someone else clones this repository on a fresh machine, they would need to obtain the same files from the Kaggle Rossmann competition and place them in data/ using the same filenames.

5\. Training the Model & Generating Results
-------------------------------------------

With the virtual environment activated and data in place, run:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python rossmann_dnn.py   `

What rossmann\_dnn.py does:

1.  **Load data**
    
    *   Reads train.csv, test.csv, and store.csv.
        
    *   Merges store metadata into train/test.
        
2.  **Feature engineering**
    
    *   Date-based features: Year, Month, Day, WeekOfYear.
        
    *   Competition features: HasCompetition, CompetitionDistance (imputed), CompetitionOpenMonths.
        
    *   Promo2 features: Promo2OpenWeeks, IsPromoMonth.
        
    *   Per-store lag & rolling features based on Sales.
        
    *   Store-level aggregates: StoreAvgSales, StoreMedianSales, StoreDowAvgSales.
        
    *   Filters to **open days with positive sales** in the training set.
        
3.  **Train/Validation split**
    
    *   Time-based split:
        
        *   Train: Date < 2015-06-15
            
        *   Validation: Date ≥ 2015-06-15
            
    *   Target: y = log1p(Sales).
        
4.  **Preprocessing & model**
    
    *   Uses ColumnTransformer:
        
        *   StandardScaler on numeric features.
            
        *   OneHotEncoder(handle\_unknown="ignore") on categorical features.
            
    *   Builds a Deep Neural Network with TensorFlow/Keras:
        
        *   Dense(512) → BatchNorm → Dropout(0.3)
            
        *   Dense(256) → BatchNorm → Dropout(0.3)
            
        *   Dense(128) → Dropout(0.2)
            
        *   Output Dense(1, linear) for log1p(Sales).
            
    *   Optimizer: Adam (learning\_rate=1e-3).
        
    *   Loss: MSE on log1p(Sales).
        
    *   Metric: custom RMSPE in original sales space.
        
    *   Uses EarlyStopping and ReduceLROnPlateau on validation RMSPE.
        
5.  **Training output**
    
    *   Prints training / validation metrics each epoch.
        
    *   Validation RMSPE:
        
6.  **Test predictions & submission**
    
    *   Applies the trained model to test.csv.
        
    *   Inverse-transforms predictions back to Sales using np.expm1.
        
    *   Sets Sales = 0 for closed days (Open == 0).
        
    *   Clamps any negative predictions to zero.
        
    *   submission.csv
        

This submission.csv can be uploaded directly to Kaggle.

6\. Generating Plots (Optional, for Report / Analysis)
------------------------------------------------------

The notebook rossmann\_plots.ipynb is used to generate diagnostic figures, such as:

*   **Predicted vs Actual Sales (Validation)**
    
*   **Example Store – Actual vs Predicted Sales over Time**
    

Typical workflow:

1.  Open rossmann\_plots.ipynb in VS Code / Jupyter.
    
2.  Select the venv Python kernel.
    
3.  Run all cells.
    

The notebook:

*   Rebuilds the dataset and features (same as rossmann\_dnn.py).
    
*   Trains a model on the time-based split.
    
*   Saves plots into the figures/ directory, e.g.:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   figures/  ├── pred_vs_actual_val.png  └── store__val_actual_vs_pred.png   `

You can use these plots directly in your assignment report.

7\. Reproducibility Notes
-------------------------

*   np.random.seed(42)tf.random.set\_seed(42)
    
*   However, due to multi-threading and low-level optimizations, deep learningtraining can still be slightly non-deterministic.As a result, the **exact** validation RMSPE may vary between runs(for example, around **0.12–0.16**).
    
*   python rossmann\_dnn.py
    

8\. Acknowledgements
--------------------

*   Dataset and competition: **Kaggle – Rossmann Store Sales**
    
*   Implementation developed as part of a **Deep Learning assignment**focusing on time-series forecasting with Deep Neural Networks.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   If you want an even shorter “assignment-only” README (like 1–2 sections), I can compress this down too.  ::contentReference[oaicite:0]{index=0}   `
