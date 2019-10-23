#
# OtterTune - forms.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from django import forms

from .db import target_objectives
from .models import Session, Project, Hardware, SessionKnob
from .types import StorageType


class NewResultForm(forms.Form):
    upload_code = forms.CharField(max_length=30)
    metrics_before = forms.FileField()
    metrics_after = forms.FileField()
    knobs = forms.FileField()
    summary = forms.FileField()


class ProjectForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        self.user_id = kwargs.pop('user_id')
        self.project_id = kwargs.pop('project_id')
        super().__init__(*args, **kwargs)

    def is_valid(self):
        valid = super().is_valid()
        if valid:
            new_name = self.cleaned_data['name']
            user_projects = Project.objects.filter(
                user__id=self.user_id, name=new_name)
            if self.project_id:
                user_projects = user_projects.exclude(id=self.project_id)
            if user_projects.exists():
                valid = False
                self._errors['name'] = ["Project '{}' already exists.".format(new_name)]
        return valid

    class Meta:  # pylint: disable=no-init
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

    cpu = forms.IntegerField(label='Number of CPUs', min_value=1)
    memory = forms.IntegerField(label='Memory (GB)', min_value=1)
    storage = forms.IntegerField(label='Storage (GB)', min_value=1)
    storage_type = forms.ChoiceField(label='Storage Type', choices=StorageType.choices())

    def __init__(self, *args, **kwargs):
        self.project_id = kwargs.pop('project_id')
        self.user_id = kwargs.pop('user_id')
        self.session_id = kwargs.pop('session_id')

        super().__init__(*args, **kwargs)
        self.fields['description'].required = False
        self.fields['target_objective'].required = True
        self.fields['tuning_session'].required = True
        self.initial.update(cpu=4, memory=16, storage=32,
                            storage_type=StorageType.SSD)

        target_objs = target_objectives.get_all()
        choices = set()
        for entry in target_objs.values():
            for name, obj in entry.items():
                choices.add((name, obj.label))
        target_obj_choices = sorted(choices)
        self.fields['target_objective'].widget = forms.Select(
            choices=target_obj_choices)

    def is_valid(self):
        valid = super().is_valid()
        if valid:
            new_name = self.cleaned_data['name']
            user_sessions = Session.objects.filter(
                user__id=self.user_id, project__id=self.project_id, name=new_name)
            if self.session_id:
                user_sessions = user_sessions.exclude(id=self.session_id)
            if user_sessions.exists():
                valid = False
                self._errors['name'] = ["Session '{}' already exists.".format(new_name)]

            if valid:
                dbms = self.cleaned_data['dbms']
                assert dbms is not None
                target_obj_name = self.cleaned_data['target_objective']
                try:
                    target_objectives.get_instance(dbms.pk, target_obj_name)
                except KeyError:
                    self._errors['target_objective'] = \
                        ["Invalid target objective '{}' for dbms {}.".format(
                            target_obj_name, dbms.full_name)]
                    valid = False

        return valid

    def save(self, commit=True):
        model = super().save(commit=False)

        cpu2 = self.cleaned_data['cpu']
        memory2 = self.cleaned_data['memory']
        storage2 = self.cleaned_data['storage']
        storage_type2 = self.cleaned_data['storage_type']

        hardware, _ = Hardware.objects.get_or_create(cpu=cpu2,
                                                     memory=memory2,
                                                     storage=storage2,
                                                     storage_type=storage_type2)

        model.hardware = hardware

        if commit:
            model.save()

        return model

    class Meta:  # pylint: disable=no-init
        model = Session

        fields = ('name', 'description', 'tuning_session', 'dbms', 'cpu', 'memory', 'storage',
                  'algorithm', 'target_objective')

        widgets = {
            'name': forms.TextInput(attrs={'required': True}),
            'description': forms.Textarea(attrs={'maxlength': 500, 'rows': 5}),
        }
        labels = {
            'dbms': 'DBMS',
        }


class SessionKnobForm(forms.ModelForm):
    name = forms.CharField(max_length=128)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['session'].required = False
        self.fields['knob'].required = False
        self.fields['name'].widget.attrs['readonly'] = True

    class Meta:  # pylint: disable=no-init
        model = SessionKnob
        fields = ['session', 'knob', 'minval', 'maxval', 'tunable']
