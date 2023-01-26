import math

class User(object):

    def __init__(self, gender, age, weight, height, activity_level, health_condition):
        self.gender = gender
        self.age = age
        self.weight = weight 
        self.height = height
        self.activity_level = activity_level 
        self.health_condition = health_condition

    # Calculate BMI - Body Mass Index
    def calculate_bmi(self):
        return self.weight / (self.height/100) ** 2  

    '''
    BMI Categories:
    Underweight = <18.5
    Normal weight = 18.5–24.9
    Overweight = 25–29.9
    Obesity = BMI of 30 or greater
    '''
    def get_bmi_category(self):
        bmi = self.calculate_bmi()

        if bmi <= 18.5:
            return "Underweight"
        elif bmi <= 24.9:
            return "Normal Weight"
        elif bmi <= 29.9:
            return "Overweight"
        else:
            return "Obese"

    # Calculate BMR - Basal Metabolic Rate 
    '''
    BMR (Men) = 10 * weight + 6.25 * height − 5 * age + 5 
    BMR (Women) = 10 * weight+6.25 * height − 5 * age − 161
    '''

    def calculate_bmr(self):
        base_bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age
        if self.gender == 'Male':
            return math.ceil(base_bmr + 5)
        elif self.gender == 'Female':
            return math.ceil(base_bmr - 161)
        else:
            return 0

    def get_recommended_daily_calorie(self):
        bmr = self.calculate_bmr()

        #Sedentary
        if self.activity_level == 0:
            return math.ceil(bmr * 1.2)
        
        # Lightly active (exercise 1-3 days a week)
        elif self.activity_level == 1:
            return math.ceil(bmr * 1.375)
        
        # Moderately active (exercise 3-5 days a week)
        elif self.activity_level == 2:
            return math.ceil(bmr * 1.55)
        
        # Very active (exercise 5-7 days a week)
        elif self.activity_level == 3:
            return math.ceil(bmr * 1.725)
        
        # Vigorous Exercise (twice every day)
        elif self.activity_level == 4:
            return math.ceil(bmr * 1.90)
        
        else:
            return 0