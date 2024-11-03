import streamlit as st

inputString = st.text_input("Please enter your text to check for palindrome")
newString = inputString.lower()

def checkPalindrome(newString):
    if newString == newString[::-1]:
        return("The string is a Palindrome")
    else:
        return("The string is not a Palindrome")

if st.button("Click to check if Palindrome"):
    st.write(checkPalindrome(newString))