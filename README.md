# Model
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
