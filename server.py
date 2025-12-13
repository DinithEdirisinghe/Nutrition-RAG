from mcp.server.fastmcp import FastMCP

# 1. Initialize the Server
# We call it "Nutrition Tools"
mcp = FastMCP("Nutrition Tools")

# 2. Define a Tool: BMI Calculator
# The @mcp.tool tag tells the server: "This function is a tool for the AI!"
@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    Calculates Body Mass Index (BMI) and returns the category.
    Args:
        weight_kg: Weight in kilograms.
        height_m: Height in meters.
    """
    bmi = weight_kg / (height_m ** 2)
    bmi = round(bmi, 2)

    category = ""
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"

    return f"BMI is {bmi} ({category})"

# 3. Define a Tool: BMR Calculator (Mifflin-St Jeor Equation)
@mcp.tool()
def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> str:
    """
    Calculates Basal Metabolic Rate (calories burned at rest).
    Args:
        weight_kg: Weight in kg
        height_cm: Height in cm
        age: Age in years
        gender: 'male' or 'female'
    """
    # The standard formula for BMR
    if gender.lower() == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    return f"BMR is {int(bmr)} calories/day"

# 4. Run the server
if __name__ == "__main__":
    # print("🔋 Nutrition Tools Server is running...")
    mcp.run()