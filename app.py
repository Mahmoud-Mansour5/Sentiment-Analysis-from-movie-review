from flask import Flask, request, jsonify, render_template_string
import joblib
import os
from translate import Translator

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'sentiment_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError("The model file 'sentiment_model.pkl' was not found.")

# Initialize the translator
translator = Translator(to_lang="en", from_lang="ar")

# Define a route for the default URL, which loads the form
@app.route('/')
def home():
    return render_template_string('''
           <!DOCTYPE html>
           <html>
           <head>
               <title>Sentiment Analysis</title>
               <style>
                   body { 
                       font-family: Arial, sans-serif; 
                       margin: 0; 
                       padding: 0;
                       background-image: url('Data Analytics.jpg'); 
                       background-size: cover;
                       background-position: center;
                       background-repeat: no-repeat;
                       color: #333; 
                       display: flex;
                       justify-content: center;
                       align-items: center;
                       height: 100vh;
                   }
                   h1 {
                       color: #4CAF50;
                       text-align: center;
                   }
                   textarea { 
                       width: 100%; 
                       height: 100px; 
                       font-size: 16px; 
                       border: 1px solid #ccc;
                       padding: 10px;
                       border-radius: 4px;
                   }
                   input[type="button"] { 
                       background-color: #4CAF50; 
                       color: white; 
                       padding: 10px 20px; 
                       border: none; 
                       border-radius: 4px; 
                       cursor: pointer; 
                       font-size: 16px; 
                       display: block;
                       margin: 10px auto;
                   }
                   input[type="button"]:hover { 
                       background-color: #45a049; 
                   }
                   #result { 
                       margin-top: 20px; 
                       font-weight: bold; 
                       font-size: 18px; 
                       text-align: center;
                   }
                   form {
                       background-color: rgba(255, 255, 255, 0.8);
                       padding: 20px;
                       border-radius: 8px;
                       box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                       max-width: 600px;
                       width: 100%;
                       text-align: center;
                   }
               </style>
           </head>
           <body>
               <form id="reviewForm">
                   <h1>Sentiment Analysis</h1>
                   <label for="review">Enter your movie review (in Arabic or English):</label><br>
                   <textarea id="review" name="review" rows="4" cols="50" placeholder="Enter your movie review here👇🏻👇🏻..."></textarea><br>
                   <input type="button" value="Predict Sentiment" onclick="predictSentiment()">
                   <div id="result"></div>
               </form>
               <script>
                   function predictSentiment() {
                       var review = document.getElementById("review").value;
                       fetch('/predict', {
                           method: 'POST',
                           headers: {
                               'Content-Type': 'application/json'
                           },
                           body: JSON.stringify({ review: review })
                       })
                       .then(response => response.json())
                       .then(data => {
                           document.getElementById("result").innerHTML = "Sentiment: " + data.sentiment + "<br>Translated Review: " + data.translated_review;
                       });
                   }
               </script>
           </body>
           </html>
           ''')

# Define a route for handling the sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']

    # Translate the review to English if it is in Arabic
    translated_review = translator.translate(review)

    # Predict sentiment
    prediction = model.predict([translated_review])
    sentiment = 'Your feeling is : " GOOD Feeling😇🫶🏼."' if prediction[0] == 1 else 'Your feeling is : " BAD Feeling😞🥺."'

    # Return the sentiment and the translated review
    return jsonify(sentiment=sentiment, translated_review=translated_review)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)