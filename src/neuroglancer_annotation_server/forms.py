from flask_wtf import FlaskForm
from wtforms import FieldList, BooleanField, RadioField


class NgDataSetExtensionForm(FlaskForm):
    dataset = RadioField('datasets')
    extensions = FieldList(BooleanField('extension'), label='extensions')
