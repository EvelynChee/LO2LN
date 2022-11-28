# EyePACS Dataset

* Download dataset from https://www.kaggle.com/c/diabetic-retinopathy-detection/data and extract the images into folders `train` and `test` respectively
* Run the following preprocessing script 
    ```bash
    python3 preprocess.py
    ```
* You should have the following structure:
    ```
    .
    ├── ... 
    ├── train_process            
    ├── test_process            
    ├── trainLabels_process.csv      
    ├── testLabels_process.csv       
    ├── preprocess.py       
    └── ...
    ```
