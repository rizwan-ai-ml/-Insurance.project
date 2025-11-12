# 🧠 Insurance Cost Prediction Project

This Machine Learning project predicts **medical insurance costs** based on personal and demographic features such as age, BMI, smoking status, and region.  
It uses **Linear Regression** to analyze patterns in the data and provide accurate insurance cost predictions.

---

## 📘 Project Overview

Health insurance cost estimation is a critical task for both insurance companies and individuals.  
The cost depends on several factors such as **age, gender, lifestyle habits, and BMI (Body Mass Index)**.  

This project uses **Supervised Learning** to train a model that learns from historical data and predicts the approximate insurance charge for a person.

---

## 🎯 Objective
- To analyze the relationship between different features (age, BMI, smoker, region, etc.) and insurance cost.  
- To build a **Linear Regression model** that can predict the insurance cost for a new individual.  
- To visualize the data and understand how different attributes impact the insurance cost.

---

## 🧩 What’s Used in This Project (UG/Used)

### 💻 Programming Language:
- **Python**

### 🧰 Tools & Platforms:
- **Google Colab** → For development and running the notebook  
- **Jupyter Notebook (.ipynb)** → Project format  
- **GitHub** → For version control and hosting the project

### 📚 Python Libraries Used:
| Library | Purpose |
|----------|----------|
| **NumPy** | For numerical operations and array handling |
| **Pandas** | For data manipulation, cleaning, and analysis |
| **Matplotlib** | For creating graphs and visualizations |
| **Seaborn** | For advanced visualization and correlation heatmaps |
| **Scikit-learn (sklearn)** | For training, testing, and evaluating the ML model |

### 🧠 Machine Learning Algorithm:
- **Linear Regression** (from `sklearn.linear_model`)  
  Used for predicting continuous output values (insurance cost).

---

## 📊 Dataset Information

The dataset contains customer information with the following columns:

| Feature | Description |
|----------|--------------|
| `age` | Age of the insured person |
| `sex` | Gender (male/female) |
| `bmi` | Body Mass Index (ratio of weight to height) |
| `children` | Number of dependents covered by insurance |
| `smoker` | Whether the person is a smoker or not |
| `region` | Residential area (northeast, northwest, southeast, southwest) |
| `charges` | Final insurance cost — this is the **target variable** |

---

## ⚙️ Project Workflow

### Step 1️⃣: Importing Libraries
All essential Python libraries are imported for analysis and modeling.

### Step 2️⃣: Loading the Dataset
The dataset is read using `pandas.read_csv()` and the first few rows are displayed using `head()`.

### Step 3️⃣: Data Analysis & Visualization
- Checked missing values using `isnull().sum()`  
- Visualized feature distributions using **Seaborn**  
- Created correlation heatmap to understand feature relationships

### Step 4️⃣: Data Preprocessing
- Converted categorical variables (like smoker, sex, region) into numeric format using **Label Encoding** or **OneHotEncoding**.  
- Split the dataset into **training and testing sets** using `train_test_split()`.

### Step 5️⃣: Model Training
- Implemented **Linear Regression model** from `sklearn.linear_model`.  
- Trained the model on the training dataset using `fit()` function.

### Step 6️⃣: Model Prediction
- Used the trained model to predict insurance costs on the test dataset.  
- Compared predicted values with actual values.

### Step 7️⃣: Model Evaluation
- Calculated performance metrics:  
  - **R² Score** – To measure model accuracy  
  - **Mean Squared Error (MSE)** – To measure prediction error  

---

## 📈 Results & Insights

- The model successfully learned the relationship between features and insurance costs.  
- It shows a strong correlation between **smoker status** and **charges** — smokers generally pay higher premiums.  
- BMI and age also significantly impact insurance cost.  
- The **R² Score** indicates a good fit of the model with the dataset.

---

## 🚀 Future Improvements

- Try **Polynomial Regression** or **Random Forest Regression** for higher accuracy.  
- Add more real-world data for better generalization.  
- Deploy the model using **Flask** or **Streamlit** to make it a web app.  
- Build an **interactive UI** where users can input details and get instant predictions.

---

## 📂 File Structure
Insurance.project/
│
├── Copy_of_Insurance.ipynb # Main Jupyter Notebook file
├── README.md # Project documentation
└── dataset.csv (optional) # Dataset file (if uploaded)


---

## 👨‍💻 Author

**Rizwan**  
AI/ML Developer | Python Programmer | Data Science Learner  

📧 rizwanalam707040@gmail.com  
🌐 GitHub: [https://github.com/rizwan-ai-ml](https://github.com/rizwan-ai-ml)

---

⭐ *If you found this project helpful, please give it a star on GitHub!*  
