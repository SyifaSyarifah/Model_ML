# Machine Learning Documentation
### From Team Giziwise

The model we created is a nutrition prediction machine learning, using linear regression for its concept. With the theme of Nutrition, our team presents the predicted nutritional values based on the 2017 Indonesian Food Composition Table data as TKPI 2017.


# Project Planning and Setup
Goals:Objective Calculate the nutrients entering the body, through each food input to generate nutritional prediction information.
Users: End user
Evaluation  : Using Random Forest Accuracy tuned for accuracy and validation. Predicting food through the designed modelling stage. Then produce the result of the testing model.

# Model Building's Notebook

[Notebook modeling building](https://github.com/SyifaSyarifah/Model_ML/blob/main/ML_Model_Regresi_for_Group.ipynb)


# Dataset Resources

[Dataset Nutrisi after cleaning from TKPI 2017](https://github.com/SyifaSyarifah/Model_ML/blob/main/dataset/Dataset_Giziwise.csv)


# Result Model
Model repository for machine learning division
## Predict food
- [POST] /predict

### Request body
| Parameter    | Type   | Description   |
| ------------ | ------ | ------------- |
| nama_makanan | String | Nama          |
| portion_size | float  | Portion size  |

### Response
| Parameter | Type   | Description      |
| --------- | ------ | ---------------- |
| energi    | float  | energi per porsi |
| lemak     | float  | lemak makanan    |
| protein   | float  | protein maknan   |

```
{
        "energi": 89.21,
        "lemak": 1.52,
        "protein": 1.52
    }
}
```

# How to run this Flask app with local computer
- Clone this repo
- Open terminal then use project's root directory
- Type python -m venv .venv and hit enter
- If, In Linux, type source .venv/bin/activate
- If, In Windows, type .venv\Scripts\activate
- Type pip install -r requirements.txt in terminal
- Serve the Flask app by typing flask run
- It will run on http://192.168.1.8:8080

![alt text](https://github.com/SyifaSyarifah/Model_ML/blob/main/Deploy%20Lokal.png?raw=true)

# How to predict nutrisi value with Postman
- Open Postman
- Enter URL request bar with http://192.168.1.8:8080/predict
- Select method POST
- Make sure the hearders Key section: Content-Type and Value: application/json (Type)
- Go to Body tab and select raw
Input the nama_makanan": "up to you",
  "portion_size": Up to you " that you want predict as a value of the key
Send the request

![alt text](https://github.com/SyifaSyarifah/Model_ML/blob/main/Postman.jpg?raw=true)

# Requirements
Library used in this project:

-numpy==1.25.2
-pandas==2.0.3
-scikit-learn==1.2.2
-joblib==1.4.2
-flask==2.0.3
-functions-framework==3.0.0
-werkzeug==2.0.3
-google-cloud-storage>=1.44.0

