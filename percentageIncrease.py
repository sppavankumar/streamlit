import streamlit as st

st.header("Percentage Increase Calculator")
a = st.number_input("Enter initial value: ",min_value=0, value=25,step=1)
b = st.number_input("Enter final value: ", min_value=1,value=50,step=2)

def percentageIncrease(x,y):
    return ((y-x)/x)*100

if st.button("Calculate"):
    perIncrease = percentageIncrease(a,b)
else:
    st.write("Click button to calculate")
st.write(f"The total percentage Increase is: {perIncrease:.2f} %")