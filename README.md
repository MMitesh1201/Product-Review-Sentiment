
# Product Review Sentiment

This application allows users to upload a CSV file containing product reviews and analyzes the sentiment of each review. The app processes the reviews from a specific column named **reviews.text** in the uploaded file and classifies them as either positive or negative sentiments.

**Features:**

File Upload:

You can upload a CSV file with a column labeled reviews.text, which contains the product reviews.

Sentiment Analysis:

 The app uses a pre-trained machine learning model to analyze the sentiment of each review and categorize it as either positive or negative.

Downloadable Results:

 After processing, the app generates a CSV file with a new column named sentiment, which contains the sentiment classification for each review. This file can be downloaded.
 
Sentiment Distribution Bar Chart:

 The app also displays a bar chart showing the count of positive and negative reviews, providing a visual summary of the sentiment distribution.

**How to Use:**

Upload a CSV file with the reviews.text column.
The app will process the file and classify the sentiment of each review.
Download the results as a CSV file with an added sentiment column.
View a bar graph showing the distribution of positive and negative reviews.


## Authors

- [@MMitesh1201](https://github.com/MMitesh1201)


## Running Tests

To run tests, run the following command in the command propt where the app.py file is located.

```bash
  streamlit run app.py
```


## Dependencies


Install dependencies

```bash
 streamlit 
 contractions
 nltk
 nltk.corpus 
 transformers
 nltk.tokenize
 pandas 
 plotly
```



## Tech Stack

**Models:** nlptown/bert-base-multilingual-uncased-sentiment

**Server:** Streamlit


## 🚀 About Me
🔭 I’m currently working on Sentimental Analysis

🌱 I’m currently learning Generative AI

💬 Ask me about Streamlit,Huggingface

📫 How to reach me miteshagrawal27@gmail.com


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mitesh-agrawal-9934ab215/)


