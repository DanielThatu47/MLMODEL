from main import QGen

# Initialize the QGen object
qgen = QGen()

# Take input text from the user
input_text = input("Enter the text for which you want to generate MCQs: ")

# Generate MCQs
payload = {"input_text": input_text}
output = qgen.predict_mcq(payload)

# Print the generated MCQs
print("\nGenerated MCQs:")



# Iterate through each question in the output
for index, question in enumerate(output["questions"], start=1):
    # Print the question number and content
    print(f"\nQuestion {index}:", question.get("question_statement"))
    
    # Print the generated question

    # Print the options
    options = question.get("options", [])
    for option_index, option in enumerate(options, start=1):
        print(f"Option {option_index}: {option}")
    
    # Print the correct answer
    correct_answer = question.get("answer", "Correct answer not available")
    print("Correct Answer:", correct_answer)
question_statement = question.get("question_statement")   
