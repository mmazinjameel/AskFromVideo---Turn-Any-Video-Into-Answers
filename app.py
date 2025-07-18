from flask import Flask, render_template, request, redirect, url_for, session
from RAG import RAG_Class 

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form.get('question')
        link = request.form.get('Youtube_Link')

        ## Call the Class
        rag = RAG_Class(link)  # just the video ID
        answer = rag.prompt(question)
        session['answer'] = answer
        session['question'] = question
        session['link'] = link
        return redirect(url_for('home'))

    

    answer = session.pop('answer', None)
    question = session.pop('question', '')
    link = session.pop('link','')
    return render_template("index.html", answer=answer, question=question, link=link)

if __name__ == "__main__":
    app.run(debug=True)
