#
# OtterTune - forms.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 25, 2017

@author: dvanaken
'''

from django import forms
from django.db.models import Max

from .models import Session, Project, Hardware, SessionKnob


class NewResultForm(forms.Form):
    upload_code = forms.CharField(max_length=30)
    metrics_before = forms.FileField()
    metrics_after = forms.FileField()
    knobs = forms.FileField()
    summary = forms.FileField()


class ProjectForm(forms.ModelForm):

    class Meta:  # pylint: disable=old-style-class,no-init
        model = Project

        fields = ['name', 'description']

        widgets = {
            'name': forms.TextInput(attrs={'required': True}),
            'description': forms.Textarea(attrs={'maxlength': 500,
                                                 'rows': 5}),
        }


class SessionForm(forms.ModelForm):

    gen_upload_code = forms.BooleanField(widget=forms.CheckboxInput,
                                         initial=False,
                                         required=False,
                                         label='Get new upload code')

    cpu = forms.IntegerField(label='Number of Processors')
    memory = forms.FloatField(label='RAM (GB)')
    storage = forms.IntegerField(label='Storage (GB)')

    def __init__(self, *args, **kwargs):
        super(SessionForm, self).__init__(*args, **kwargs)
        self.fields['description'].required = False
        self.fields['target_objective'].required = False
        self.fields['tuning_session'].required = True
        self.fields['cpu'].initial = 2
        self.fields['memory'].initial = 16.0
        self.fields['storage'].initial = 32

    def save(self, commit=True):
        model = super(SessionForm, self).save(commit=False)

        cpu2 = self.cleaned_data['cpu']
        memory2 = self.cleaned_data['memory']
        storage2 = self.cleaned_data['storage']

        if hasattr(model, 'hardware'):
            model.hardware.cpu = cpu2
            model.hardware.memory = memory2
            model.hardware.storage = storage2
            model.hardware.save()
        else:
            last_type = Hardware.objects.aggregate(Max('type'))['type__max']
            if last_type is None:
                last_type = 0
            model.hardware = Hardware.objects.create(type=last_type + 1,
                                                     name='New Hardware',
                                                     cpu=cpu2,
                                                     memory=memory2,
                                                     storage=storage2,
                                                     storage_type='Default',
                                                     additional_specs='{}')

        if commit:
            model.save()

        return model

    class Meta:  # pylint: disable=old-style-class,no-init
        model = Session

        fields = ('name', 'description', 'tuning_session', 'dbms', 'cpu', 'memory', 'storage',
                  'target_objective')

        widgets = {
            'name': forms.TextInput(attrs={'required': True}),
            'description': forms.Textarea(attrs={'maxlength': 500,
                                                 'rows': 5}),
        }
        labels = {
            'dbms': 'DBMS',
        }


class SessionKnobForm(forms.ModelForm):
    name = forms.CharField(max_length=128)

    def __init__(self, *args, **kwargs):
        super(SessionKnobForm, self).__init__(*args, **kwargs)
        self.fields['session'].required = False
        self.fields['knob'].required = False
        self.fields['name'].widget.attrs['readonly'] = True

    class Meta:  # pylint: disable=old-style-class,no-init
        model = SessionKnob
        fields = ['session', 'knob', 'minval', 'maxval', 'tunable']
