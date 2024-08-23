from flask import Flask, redirect, render_template, request, jsonify # type: ignore
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
import torch # type: ignore
import os

app = Flask(__name__)

# Initialize DataFrame to store reviews, sentiments, and feedback
reviews_df = pd.DataFrame(columns=["Review", "Sentiment","Delivery process", 
                       "Product Quality ", "Product Information Accuracy", "Payment Security", 
                        "Stock Availability","advice"])

# Load pre-trained BERT model and tokenizer for sentiment analysis
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("model")
bert_model.eval()

def analyze_sentiment(text):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return "Positive" if predicted_class == 2 else "Negative"

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]

    # Analyze sentiment of user's review
    sentiment = analyze_sentiment(user_input)

    # Generate response based on sentiment
    if sentiment == "Positive":  # Positive sentiment
        response = "Thank you for your positive feedback! We're thrilled to hear that you had a great experience. Your satisfaction is our top priority, and we're always here to provide you with excellent service."  # You can customize the prompt as per your requirement
    else:  # Negative sentiment
        response = "Thank you for your feedback. We're sorry to hear that you're not satisfied with your experience. Your feedback helps us improve, and we'll do our best to address your concerns and make things right."  # You can customize the prompt as per your requirement

    # Save review and sentiment to DataFrame
    global reviews_df
    if not isinstance(reviews_df, pd.DataFrame):
        reviews_df = pd.DataFrame(columns=["Review", "Sentiment"])
    reviews_df = pd.concat(
        [reviews_df, pd.DataFrame({"Review": [user_input], "Sentiment": [sentiment]})],
        ignore_index=True,
    )
    # reviews_df.to_csv("response.csv",mode='a', header=False)
    return response

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    feature1 = request.form["feature1"]
    feature2 = request.form["feature2"]
    feature3 = request.form["feature3"]
    feature4 = request.form["feature4"]
    feature5 = request.form["feature5"]
    global reviews_df
    if isinstance(reviews_df, pd.DataFrame):
        reviews_df["Delivery process"] = feature1
        reviews_df["Product Quality "] = feature2
        reviews_df["Product Information Accuracy"] = feature3
        reviews_df["Payment Security"] = feature4
        reviews_df["Stock Availability"] = feature5
        # reviews_df.to_csv("response.csv",mode='a', header=False)
    return reviews_df


@app.route("/advice", methods=["POST"])
def advice():
    advice = request.form["advice"]
    global reviews_df
    if isinstance(reviews_df, pd.DataFrame):
        # Set advice for the entire DataFrame, filling with NaN for empty rows
        reviews_df["advice"] = advice
        #reviews_df.loc[reviews_df.index[-1], "advice"] = advice  # Add advice to the last row

        # Define the column order and ensure all columns are present
        column_order = ["Review", "Sentiment", "Delivery process", 
                       "Product Quality ", "Product Information Accuracy", "Payment Security", 
                        "Stock Availability","advice"]
        for column in column_order:
            if column not in reviews_df.columns:
                reviews_df[column] = pd.NA  # Fill missing columns with NA

        # Reorder the DataFrame columns
        reviews_df = reviews_df[column_order]

        # Replace NaN values with a placeholder if needed
        reviews_df.fillna('', inplace=True)

        # Check if CSV file exists to determine whether to write headers
        csv_file_path = "static/images/response.csv"
        write_header = not os.path.exists(csv_file_path)

        # Save the last two rows to CSV
        last_two_rows = reviews_df.tail(1)
        last_two_rows.to_csv(csv_file_path, mode="a", header=write_header, index=False)

    # return advice
    return advice

@app.route("/login_action", methods=["POST"])
def login_action():
    username = request.form["username"]
    password = request.form["password"]

    if username == "admin" and password == "password":
        return redirect("/welcome")  # Redirect to the '/welcome' route
    else:
        return redirect("/login")

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

# Inside your app.py

@app.route('/get-csv')
def get_csv():
    with open('static/images/response.csv', 'r') as file:
        return file.read(), 200, {'Content-Type': 'text/csv'}



if __name__ == "__main__":
    app.run(debug=True)
