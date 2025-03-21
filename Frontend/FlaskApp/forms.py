from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import InputRequired, Length, NumberRange
from wtforms.widgets import TextArea

class GenerateTitlesForm(FlaskForm):
    abstract = StringField('Abstract', widget=TextArea(), validators=[InputRequired(), Length(min=1, max=10000)])
    num_return_sequences = IntegerField('Number of Return Sequences', validators=[InputRequired(), NumberRange(min=1, max=25)])
    temperature = IntegerField('Temperature', validators=[InputRequired(), NumberRange(min=1, max=100)])
    beam_width = IntegerField('Beam Width', validators=[InputRequired(), NumberRange(min=1, max=128)])
    submit = SubmitField('Generate Titles')

    def validate(self):
        # Call super validate method
        rv = FlaskForm.validate(self)
        if not rv:
            return False

        # Check if num_return_sequences is less than or equal to beam_width
        if self.num_return_sequences.data > self.beam_width.data:
            self.num_return_sequences.errors.append('Number of Return Sequences must be less than or equal to Beam Width')
            return False

        return True