from flask import render_template, url_for, request, flash
from wtforms.validators import NumberRange

from FlaskApp import app, csrf
from FlaskApp.forms import GenerateTitlesForm

from FlaskApp.utils import generate_titles

@app.route('/', methods=['GET', 'POST'])
def index(): 

    form = GenerateTitlesForm()

    if request.method == 'POST':
        form = GenerateTitlesForm(request.form)
        if not form.validate():
            flash('All fields are required.')
            return render_template('index.html', form=form, titles='')
        
        params = {
            'abstract': form.abstract.data,
            'num_return_sequences': form.num_return_sequences.data,
            'temperature': form.temperature.data,
            'beam_width': form.beam_width.data
        }

        titles = generate_titles(params)

        titles = '\n'.join(titles)

        return render_template('index.html', form=form, titles=titles)

    return render_template('index.html', form=form, titles='')
