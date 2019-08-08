from flask_wtf import FlaskForm
from wtforms import FieldList, BooleanField, RadioField


# class NgDataSetExtensionForm(FlaskForm):
#     dataset = RadioField('datasets')
#     extensions = FieldList(BooleanField('extension'), label='extensions')

def build_extension_forms(datasets, extensions):
    class NgDataSetExtensionForm(FlaskForm):
        dataset = RadioField('datasets')
        
    for ext in extensions:
        setattr(NgDataSetExtensionForm,
                ext,
                BooleanField(label='extension', id=ext))

    form = NgDataSetExtensionForm()
    form.dataset.choices = [(d, d) for d in datasets]
    return form
