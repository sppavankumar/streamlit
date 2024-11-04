import streamlit as st

def pctAmount(amount:float, pct:float) -> float:
    percentageAmount = (pct/100)*amount
    return amount-percentageAmount

def main():
    st.title("Determine amount after percentage adjustment")
    
    amount_input = st.number_input("Enter the total amount",min_value=1)
    pct_input = st.number_input("Enter the percentage to be applied",min_value=1)

    if st.button("Calculate"):
        if amount_input > 0 and pct_input > 0:
            finalAmount = pctAmount(amount_input,pct_input)
            st.success(f"{amount_input:.2f} adjusted by {pct_input:.2f} percent is = {finalAmount:.2f}") 
        else:
            st.warning("Please enter number greater than zero")
    
if __name__ == "__main__":
    main()