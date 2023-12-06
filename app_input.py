import streamlit as st 

######### Example 1

# educ = st.selectbox("Education level", 
#              options = ["High School Diploma",
#                         "College Degree",
#                         "Graduate Degree"])

# st.write(f"Education (pre-conversion): {educ}")

# if educ == "High School Diploma":
#     educ = 1
# elif educ == "College Degree":
#     educ = 2
# else:
#     educ = 3
    
# st.write(f"Education (post-conversion): {educ}")



########## Example 2

# num1 = st.slider(label="Enter a number", 
#           min_value=1,
#           max_value=9,
#           value=7)

# num2 = st.slider(label="Enter a number",
#           min_value=1,
#           max_value=100,
#           value=50)

# num3 = st.slider(label="Enter a number", 
#           min_value=1,
#           max_value=8,
#           value=5)

# st.write("Your numbers: ", num1, num2, num3)

# num_sum = num1+num2+num3

# st.write("Your numbers: ", num1, num2, num3, "sum to", num_sum)



########## Example 3

with st.sidebar:
    inc = st.number_input("Income (low=1 to high=9)", 1, 9)
    deg = st.number_input("College degree (no=0 to yes=1)", 0, 1)
    mar = st.number_input("Married (0=no, 1=yes)", 0, 1)

# Create labels from numeric inputs

# Income
if inc <= 3:
    inc_label = "low income"
elif inc > 3 and inc < 7:
    inc_label = "middle income"
else:
    inc_label = "high income"

# Degree   
if deg == 1:
    deg_label = "college graduate"
else:
    deg_label = "non-college graduate"
    
# Marital
if mar == 1:
    mar_label = "married"
else:
    mar_label = "non-married"

st.write(f"This person is {mar_label}, a {deg_label}, and in a {inc_label} bracket")
