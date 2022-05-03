from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def get_status():
    return render_template('index.html', location= 'kitchen', activity= 'cooking')


if __name__ == '__main__':
    app.run()