# backend.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from main import QGen

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Initialize the QGen object
qgen = QGen()

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    # Get the input text from the request
    input_text = request.json.get('input_text')

    # Generate MCQs
    payload = {"input_text": input_text}
    output = qgen.predict_mcq(payload)

    # Prepare the response
    response_data = {
        "questions": [
            {
                "question_statement": question.get("question_statement"),
                "options": question.get("options", []),
                "answer": question.get("answer", "Correct answer not available")
            }
            for question in output["questions"]
        ]
    }

    # Return the generated MCQs as JSON
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
