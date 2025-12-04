import streamlit as st
import requests

# Define the API URL
api_url = "http://127.0.0.1:8000/ask"

# Function to send a message to the API and receive the result
def get_response(message):
    headers = {'Content-Type': 'application/json'}
    data = {'message': message}
    
    # Sending the request to the API
    response = requests.post(api_url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('results', '')
    else:
        return "Error: Unable to get a response from the API."

# Streamlit app layout
def main():
    st.title("Chatbot Prototype 💬")

    # Session state to store chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display previous messages
    for message in st.session_state['messages']:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Display user message
        st.session_state['messages'].append({'role': 'user', 'content': user_input})
        st.chat_message('user').markdown(user_input)
        
        # Get the response from the API
        with st.spinner("Thinking..."):
            bot_response = get_response(user_input)
        
        # Display bot response
        st.session_state['messages'].append({'role': 'assistant', 'content': bot_response})
        st.chat_message('assistant').markdown(bot_response)

if __name__ == "__main__":
    main()
