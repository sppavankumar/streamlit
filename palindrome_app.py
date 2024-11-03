import streamlit as st

def checkPalindrome(inputText: str) -> bool:
    cleaned_text = inputText.lower().replace(" ","")
    return cleaned_text==cleaned_text[::-1]

def main():
    st.title("Palindrome Checking App")

    userInput = st.text_input("Enter text to check palindrome")
    if st.button("Click to Check"):
        if userInput.strip():
            result = checkPalindrome(userInput)
            if result:
                st.success("The string is a palindrome") 
            else:
                st.warning("The string is not a palindrome")
        else:
            st.error("Please enter text to check")

if __name__ == "__main__":
    main()