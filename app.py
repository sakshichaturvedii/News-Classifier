from flask import Flask, render_template, request  
import pickle 

app = Flask(__name__)  

with open('model/fake_news_model.pkl', 'rb') as f:
    fake_model = pickle.load(f)

with open('model/fake_news_vectorizer.pkl', 'rb') as f:
    fake_vectorizer = pickle.load(f)

with open('model/topic_model.pkl', 'rb') as f:
    topic_model = pickle.load(f)

with open('model/topic_vectorizer.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  
def predict():
    text = request.form['news_text']
    text_clean = text.lower()

    topic_vec = topic_vectorizer.transform([text_clean])

    topic_pred = topic_model.predict(topic_vec)[0]  

    topic_proba = topic_model.predict_proba(topic_vec)[0]  
    topic_confidence = max(topic_proba) * 100  


    fake_vec = fake_vectorizer.transform([text_clean])
    fake_pred = fake_model.predict(fake_vec)[0]  

    fake_proba = fake_model.predict_proba(fake_vec)[0]
    fake_confidence = max(fake_proba) * 100  

    return render_template(
        'result.html',
        input_text=text,                      
        topic=topic_pred,                     
        topic_confidence=round(topic_confidence, 2),  
        fake_label=fake_pred,                 
        fake_confidence=round(fake_confidence, 2)     
    )

if __name__ == '__main__':
    app.run(debug=True)  
