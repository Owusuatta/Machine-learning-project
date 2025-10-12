
# 🧠 AI Fitness Calorie Burn Prediction Model

![Fitness AI Banner](https://img.shields.io/badge/Project-Fitness%20AI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange?style=for-the-badge&logo=scikit-learn)
![Google Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-red?style=for-the-badge&logo=googlecolab)

---

## 🏋️‍♀️ Project Overview
This project uses **Machine Learning** to predict the **calorie burn intensity (Low, Medium, High)** of an individual based on their **lifestyle, fitness, and diet data**.

The goal is to help users and fitness trainers understand **how workouts, nutrition, and body composition** affect calorie burn — empowering smarter workout planning and diet optimization.

---

## ⚙️ Project Workflow

1. **📂 Data Loading & Cleaning**
   - Loaded dataset from Kaggle: [Life Style Data](https://www.kaggle.com/datasets/jockeroika/life-style-data)
   - Removed missing values and standardized categorical labels.

2. **🧩 Feature Engineering**
   - Categorical and numerical columns handled separately.
   - Used **OneHotEncoding** and **Frequency Encoding** for categorical features.
   - Normalized numeric features using `StandardScaler`.

3. **📊 Data Splitting**
   - 80% Training, 10% Validation, 10% Testing.

4. **🤖 Model Training**
   - Trained a **Logistic Regression** classifier as baseline.
   - Tuned and validated using cross-validation metrics.

5. **💾 Model Saving**
   - Saved using `joblib`:
     - `final_model.joblib`
     - `lifestyle_preprocessor.joblib`

6. **🧠 Model Testing**
   - Built an **interactive testing interface** in Google Colab using sliders & dropdowns.

---

## 🧬 Dataset Description

| Feature | Description |
|----------|-------------|
| **Age** | Age of the participant (years) |
| **Gender** | Male/Female |
| **Weight (kg)** | Weight in kilograms |
| **Height (m)** | Height in meters |
| **Max_BPM** | Maximum heart rate recorded |
| **Avg_BPM** | Average heart rate maintained |
| **Resting_BPM** | Resting heart rate before exercise |
| **Session_Duration (hours)** | Workout session duration |
| **Calories_Burned** | Total calories burned |
| **Workout_Type** | Type of workout (HIIT, Cardio, Strength, etc.) |
| **Fat_Percentage** | Body fat percentage |
| **Water_Intake (liters)** | Daily water intake |
| **Workout_Frequency (days/week)** | Number of workout days per week |
| **Experience_Level** | Fitness level (1 = Beginner, 2 = Intermediate, 3 = Advanced) |
| **BMI** | Body Mass Index |
| **Daily meals frequency** | Meals consumed per day |
| **Carbs / Proteins / Fats** | Daily macronutrient intake (grams) |
| **Calories** | Total calorie intake per day |
| **meal_name / meal_type / diet_type** | Meal characteristics |
| **sugar_g / sodium_mg / cholesterol_mg** | Nutrition breakdown |
| **serving_size_g / cooking_method / prep_time_min / cook_time_min** | Meal preparation details |
| **Name of Exercise / Sets / Reps / Benefit** | Exercise details |
| **Burns Calories (per 30 min)** | Estimated calorie burn for 30 minutes |
| **Target Muscle Group / Equipment Needed / Difficulty Level / Body Part / Type of Muscle / Workout** | Exercise attributes |

---

## 🧠 Model Details

| Component | Description |
|------------|--------------|
| **Preprocessor** | `lifestyle_preprocessor.joblib` — performs scaling and encoding |
| **Model** | `final_model.joblib` — trained Logistic Regression classifier |
| **Target Variable** | `Burns_Calories_Bin` (Low, Medium, High) |

---

## 🚀 How to Run in Google Colab

### 1️⃣ Upload your files
Upload these to your Colab environment:
- `final_model.joblib`
- `lifestyle_preprocessor.joblib`

### 2️⃣ Test with sample input

```python
import joblib
import pandas as pd

# Load the preprocessor and model
preprocessor = joblib.load("lifestyle_preprocessor.joblib")
model = joblib.load("final_model.joblib")

# Create a sample input
sample = pd.DataFrame([{
    "Age": 28,
    "Gender": "Male",
    "Weight (kg)": 75,
    "Height (m)": 1.8,
    "Session_Duration (hours)": 1.2,
    "Workout_Type": "Cardio",
    "Experience_Level": "Intermediate",
    "Avg_BPM": 140,
    "Water_Intake (liters)": 2.5
}])

# Preprocess and predict
X_prep = preprocessor.transform(sample)
pred = model.predict(X_prep)[0]
print(f"🔥 Predicted Calorie Burn Level: {pred}")
````

### ✅ Example Output:

```
🔥 Predicted Calorie Burn Level: Medium
```

---

## 📦 Repository Structure

```
📁 Fitness-Calorie-Burn-Prediction/
│
├── 🧠 Fitness_Calorie_Burn_Prediction.ipynb     # Full training notebook
├── 📊 final_model.joblib                        # Trained ML model
├── ⚙️ lifestyle_preprocessor.joblib              # Preprocessing pipeline
├── 📘 README.md                                 # Project documentation
```

---

## 📈 Results Summary

| Metric                    | Value                   |
| ------------------------- | ----------------------- |
| **Accuracy (Validation)** | ~90%                    |
| **F1-Score**              | 0.88                    |
| **Precision / Recall**    | Balanced across classes |

---

## 🧩 Future Improvements

* Add deep learning models (e.g., XGBoost, RandomForest)
* Build web dashboard using Streamlit or Gradio
* Integrate with smartwatch sensor data
* Deploy model as REST API with Flask or FastAPI

---

## 👨‍💻 Author

**[Your Name]**
📧 Email: owusuatta884@gmail.com
💬 “Data + Fitness = Smarter Health Decisions 🧘‍♂️”

-

---

## ⭐ Support

If you like this project, please ⭐ star the repository and share it with other data enthusiasts
