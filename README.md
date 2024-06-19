# Machine Learning Documentation
### From Team Giziwise
---

The model we created is a nutrition prediction machine learning, using linear regression for its concept. With the theme of Nutrition, our team presents the predicted nutritional values based on the 2017 Indonesian Food Composition Table data as TKPI 2017.

---

# 1. Project Planning and Setup
Goals:Objective Calculate the nutrients entering the body, through each food input to generate nutritional prediction information.
Users: End user
Evaluation  : Using Random Forest Accuracy tuned for accuracy and validation. Predicting food through the designed modelling stage. Then produce the result of the testing model.

---
# Model Building's Notebook


---

# Dataset Resources


---

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

![alt text](?raw=true)