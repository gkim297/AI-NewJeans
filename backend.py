from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics():
    # Get seed text from request
    seed_text = request.json['seed_text']
    
    # Run the Python script and capture output
    output = subprocess.check_output(['python', 'lyrics_generator_script.py', seed_text])
    
    # Process the output if needed
    generated_lyrics = output.decode('utf-8')
    
    return jsonify({'generated_lyrics': generated_lyrics})

if __name__ == '__main__':
    app.run(debug=True)