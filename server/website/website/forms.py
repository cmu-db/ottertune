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

from .models import Session, Project, MetricCatalog


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

    def __init__(self, *args, **kwargs):
        super(SessionForm, self).__init__(*args, **kwargs)
        self.fields['description'].required = False
        self.fields['target_objective'].required = False

    def clean(self):
        super(SessionForm, self).clean()

        if not self.is_valid():
           return self.cleaned_data

        tuning_session = self.cleaned_data['tuning_session']
        target_objective = self.cleaned_data['target_objective']
        if tuning_session and not target_objective:
            raise forms.ValidationError('Please, inform a target objective \
                                         if tuning is session is enabled')

        if 'dbms' in self.cleaned_data:
            dbms = self.cleaned_data['dbms']
            metric_meta = MetricCatalog.objects.get_metric_meta(dbms, True)

            if tuning_session and target_objective not in metric_meta:
                raise forms.ValidationError('%s is an invalid target objective' % target_objective)

    class Meta:  # pylint: disable=old-style-class,no-init
        model = Session

        fields = ('name', 'description', 'tuning_session', 'dbms', 'hardware', 'target_objective')

        widgets = {
            'name': forms.TextInput(attrs={'required': True}),
            'description': forms.Textarea(attrs={'maxlength': 500,
                                                 'rows': 5}),
            'tuning_session': forms.CheckboxInput(attrs={'onclick': 'show_content();'}),
        }
        labels = {
            'dbms': 'DBMS',
        }
