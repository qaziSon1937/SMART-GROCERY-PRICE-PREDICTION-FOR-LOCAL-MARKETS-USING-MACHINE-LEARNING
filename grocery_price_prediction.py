# =====================================
# Smart Grocery Price Prediction System
# =====================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2️⃣ Load Dataset
data = pd.read_csv("weekly_grocery_prices_pakistan.csv")

print("Dataset Loaded Successfully")
print(data.head())

# 3️⃣ Convert Week to Numeric
data["Week_Number"] = data["Week"].apply(lambda x: int(x.split("-")[1]))

# 4️⃣ Encode Categorical Columns
item_encoder = LabelEncoder()
season_encoder = LabelEncoder()

data["Item"] = item_encoder.fit_transform(data["Item"])
data["Season"] = season_encoder.fit_transform(data["Season"])

# 5️⃣ Select Features & Target
X = data[["Week_Number", "Item", "Season", "Demand_Index"]]
y = data["Weekly_Avg_Price_PKR"]

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# 7️⃣ Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Model Evaluation
print("\n📊 MODEL PERFORMANCE")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²   :", r2_score(y_test, y_pred))

# 🔟 INTERACTIVE PREDICTION (USER INPUT)
print("\n🛒 GROCERY PRICE PREDICTION SYSTEM")

# Get all items from dataset
all_items = sorted(item_encoder.classes_)

# Display items nicely
print("\nAvailable Items:")
for i, item in enumerate(all_items, start=1):
    print(f"{i:2d}. {item}")
print()

# Function to safely get integer input
def get_int_input(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"⚠️  Enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("⚠️  Invalid input! Please enter a valid integer.")

# Function to safely get float input
def get_float_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"⚠️  Enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("⚠️  Invalid input! Please enter a valid number.")

# Function to safely get season input
def get_season_input(prompt):
    valid_seasons = ["Winter", "Summer", "Spring", "Autumn"]
    while True:
        value = input(prompt).strip().title()
        if value in valid_seasons:
            return value
        else:
            print(f"⚠️  Invalid season! Choose from: {', '.join(valid_seasons)}")

# Get user inputs safely
item_choice = get_int_input(f"Select item number (1–{len(all_items)}): ", 1, len(all_items))
item_name = all_items[item_choice - 1]

week_number = get_int_input("Enter week number (1–52): ", 1, 52)
season_name = get_season_input("Enter season (Winter, Summer, Spring, Autumn): ")
demand_index = get_float_input("Enter demand index (0.5 – 1.0): ", 0.5, 1.0)

# Encode inputs
item_encoded = item_encoder.transform([item_name])[0]
season_encoded = season_encoder.transform([season_name])[0]

# Prepare DataFrame
sample_input = pd.DataFrame({
    "Week_Number": [week_number],
    "Item": [item_encoded],
    "Season": [season_encoded],
    "Demand_Index": [demand_index]
})

# Predict
predicted_price = model.predict(sample_input)

# Show result
print("\n📈 PREDICTION RESULT")
print(f"Item           : {item_name}")
print(f"Week Number    : {week_number}")
print(f"Season         : {season_name}")
print(f"Demand Index   : {demand_index}")
print(f"Predicted Price: {round(predicted_price[0], 2)} PKR")



