#
# OtterTune - views.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# pylint: disable=too-many-lines
import base64
import csv
import logging
import os
import re
import shutil
import socket
import time
from collections import OrderedDict
from io import StringIO

import celery
from celery import chain, signature, uuid
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.forms import PasswordChangeForm
from django.core.exceptions import FieldError, ObjectDoesNotExist
from django.core.files.base import ContentFile, File
from django.core.management import call_command
from django.db import connection
from django.forms.models import model_to_dict
from django.http import HttpResponse, QueryDict
from django.shortcuts import redirect, render, get_object_or_404
from django.template.context_processors import csrf
from django.template.defaultfilters import register
from django.urls import reverse, reverse_lazy
from django.utils.datetime_safe import datetime
from django.utils.timezone import now
from django.views.decorators.csrf import csrf_exempt
from pytz import timezone

from . import models as app_models
from . import utils
from .db import parser, target_objectives
from .forms import NewResultForm, ProjectForm, SessionForm, SessionKnobForm
from .models import (BackupData, DBMSCatalog, ExecutionTime, Hardware, KnobCatalog, KnobData,
                     MetricCatalog, MetricData, PipelineRun, Project, Result, Session,
                     SessionKnob, User, Workload, PipelineData)
from .tasks import train_ddpg
from .types import (DBMSType, KnobUnitType, MetricType,
                    TaskType, VarType, WorkloadStatusType, AlgorithmType, PipelineTaskType)
from .utils import (JSONUtil, LabelUtil, MediaUtil, TaskUtil)
from .settings import LOG_DIR, TIME_ZONE, CHECK_CELERY

from .set_default_knobs import set_default_knobs

LOG = logging.getLogger(__name__)


# For the html template to access dict object
@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)


def signup_view(request):
    if request.user.is_authenticated():
        return redirect(reverse('home_projects'))
    if request.method == 'POST':
        post = request.POST
        form = UserCreationForm(post)
        if form.is_valid():
            form.save()
            new_post = QueryDict(mutable=True)
            new_post.update(post)
            new_post['password'] = post['password1']
            request.POST = new_post
            return login_view(request)
        else:
            LOG.warning("Signup form is not valid: %s", str(form.errors))
    else:
        form = UserCreationForm()
    token = {}
    token.update(csrf(request))
    token['form'] = form
    return render(request, 'signup.html', token)


def change_password_view(request):
    if not request.user.is_authenticated():
        return redirect(reverse('home_project'))
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            return redirect(reverse('home_projects'))
    else:
        form = PasswordChangeForm(request.user)
    token = {}
    token.update(csrf(request))
    token['form'] = form
    return render(request, 'change_password.html', token)


def login_view(request):
    if request.user.is_authenticated():
        return redirect(reverse('home_projects'))
    if request.method == 'POST':
        post = request.POST
        form = AuthenticationForm(None, post)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(reverse('home_projects'))
        else:
            LOG.warning("Login form is not valid: %s", str(form.errors))
    else:
        form = AuthenticationForm()
    token = {}
    token.update(csrf(request))
    token['form'] = form
    return render(request, 'login.html', token)


@login_required(login_url=reverse_lazy('login'))
def logout_view(request):
    logout(request)
    return redirect(reverse('login'))


@login_required(login_url=reverse_lazy('login'))
def redirect_home(request):  # pylint: disable=unused-argument
    return redirect(reverse('home_projects'))


@login_required(login_url=reverse_lazy('login'))
def home_projects_view(request):
    form_labels = Project.get_labels()
    form_labels.update(LabelUtil.style_labels({
        'button_create': 'create a new project',
        'button_delete': 'delete selected projects',
    }))
    form_labels['title'] = 'Your Projects'
    projects = Project.objects.filter(user=request.user)
    show_descriptions = any([proj.description for proj in projects])
    context = {
        "projects": projects,
        "labels": form_labels,
        "show_descriptions": show_descriptions
    }
    context.update(csrf(request))
    return render(request, 'home_projects.html', context)


@login_required(login_url=reverse_lazy('login'))
def create_or_edit_project(request, project_id=''):
    form_kwargs = dict(user_id=request.user.pk, project_id=project_id)
    if request.method == 'POST':
        if project_id == '':
            form = ProjectForm(request.POST, **form_kwargs)
            if not form.is_valid():
                return render(request, 'edit_project.html', {'form': form})
            project = form.save(commit=False)
            project.user = request.user
            ts = now()
            project.creation_time = ts
            project.last_update = ts
        else:
            project = get_object_or_404(Project, pk=project_id, user=request.user)
            form_kwargs.update(instance=project)
            form = ProjectForm(request.POST, **form_kwargs)
            if not form.is_valid():
                return render(request, 'edit_project.html', {'form': form})
            project.last_update = now()

        project.save()
        return redirect(reverse('project_sessions', kwargs={'project_id': project.pk}))
    else:
        if project_id == '':
            project = None
            form = ProjectForm(**form_kwargs)
        else:
            project = Project.objects.get(pk=project_id)
            form_kwargs.update(instance=project)
            form = ProjectForm(**form_kwargs)
        context = {
            'project': project,
            'form': form,
        }
        return render(request, 'edit_project.html', context)


@login_required(login_url=reverse_lazy('login'))
def delete_project(request):
    pids = request.POST.getlist('projects', [])
    Project.objects.filter(pk__in=pids, user=request.user).delete()
    return redirect(reverse('home_projects'))


@login_required(login_url=reverse_lazy('login'))
def project_sessions_view(request, project_id):
    sessions = Session.objects.filter(project=project_id)
    project = Project.objects.get(pk=project_id)
    form_labels = Session.get_labels()
    form_labels.update(LabelUtil.style_labels({
        'button_delete': 'delete selected session',
        'button_create': 'create a new session',
    }))
    form_labels['title'] = "Your Sessions"
    for session in sessions:
        session.session_type_name = Session.TUNING_OPTIONS[session.tuning_session]
        session.algorithm_name = AlgorithmType.name(session.algorithm)

    context = {
        "sessions": sessions,
        "project": project,
        "labels": form_labels,
    }
    context.update(csrf(request))
    return render(request, 'project_sessions.html', context)


@login_required(login_url=reverse_lazy('login'))
def session_view(request, project_id, session_id):
    project = get_object_or_404(Project, pk=project_id)
    session = get_object_or_404(Session, pk=session_id)

    # All results from this session
    results = Result.objects.filter(session=session)

    # Group the session's results by DBMS & workload
    dbmss = {}
    workloads = {}
    dbmss_ids = set()
    workloads_ids = set()
    for res in results:
        if res.dbms_id not in dbmss_ids:
            dbmss_ids.add(res.dbms_id)
            res_dbms = res.dbms
            dbmss[res_dbms.key] = res_dbms

        if res.workload_id not in workloads_ids:
            workloads_ids.add(res.workload_id)
            res_workload = res.workload
            workloads[res_workload.name] = set()
            workloads[res_workload.name].add(res_workload)

    # Sort so names will be ordered in the sidebar
    workloads = OrderedDict([(k, sorted(list(v))) for
                             k, v in sorted(workloads.items())])
    dbmss = OrderedDict(sorted(dbmss.items()))

    if len(workloads) > 0:
        # Set the default workload to whichever is first
        default_workload, default_confs = next(iter(list(workloads.items())))
        default_confs = ','.join([str(c.pk) for c in default_confs])
    else:
        # Set the default to display nothing if there are no results yet
        default_workload = 'show_none'
        default_confs = 'none'

    default_metrics = [session.target_objective]
    metric_meta = target_objectives.get_metric_metadata(
        session.dbms.pk, session.target_objective)

    knobs = SessionKnob.objects.get_knobs_for_session(session)
    knob_names = [knob["name"] for knob in knobs if knob["tunable"]]

    session.session_type_name = Session.TUNING_OPTIONS[session.tuning_session]
    session.algorithm_name = AlgorithmType.name(session.algorithm)

    form_labels = Session.get_labels()
    form_labels['title'] = "Session Info"

    context = {
        'project': project,
        'dbmss': dbmss,
        'workloads': workloads,
        'results_per_page': [10, 50, 100, 500, 1000],
        'default_dbms': session.dbms.key,
        'default_results_per_page': 10,
        'default_equidistant': "on",
        'default_workload': default_workload,
        'defaultspe': default_confs,
        'metrics': list(metric_meta.keys()),
        'metric_meta': metric_meta,
        'default_metrics': default_metrics,
        'knob_names': knob_names,
        'filters': [],
        'session': session,
        'results': results,
        'labels': form_labels,
    }
    context.update(csrf(request))
    return render(request, 'session.html', context)


@login_required(login_url=reverse_lazy('login'))
def create_or_edit_session(request, project_id, session_id=''):
    project = get_object_or_404(Project, pk=project_id, user=request.user)
    form_kwargs = dict(user_id=request.user.pk, project_id=project_id, session_id=session_id)
    if request.method == 'POST':
        if not session_id:
            # Create a new session from the form contents
            form = SessionForm(request.POST, **form_kwargs)
            if not form.is_valid():
                return render(request, 'edit_session.html',
                              {'project': project, 'form': form, 'session': None})
            session = form.save(commit=False)
            session.user = request.user
            session.project = project
            ts = now()
            session.creation_time = ts
            session.last_update = ts
            session.upload_code = MediaUtil.upload_code_generator()
            session.save()
            set_default_knobs(session)
        else:
            # Update an existing session with the form contents
            session = Session.objects.get(pk=session_id)
            form_kwargs.update(instance=session)
            form = SessionForm(request.POST, **form_kwargs)
            if not form.is_valid():
                return render(request, 'edit_session.html',
                              {'project': project, 'form': form, 'session': session})
            if form.cleaned_data['gen_upload_code'] is True:
                session.upload_code = MediaUtil.upload_code_generator()
            session.last_update = now()
            form.save()
            session.save()
        return redirect(reverse('session', kwargs={'project_id': project_id,
                                                   'session_id': session.pk}))
    else:
        if session_id:
            # Return a pre-filled form for editing an existing session
            session = Session.objects.get(pk=session_id)
            form_kwargs.update(instance=session)
            form = SessionForm(**form_kwargs)
        else:
            # Return a new form with defaults for creating a new session
            session = None
            form_kwargs.update(
                initial={
                    'dbms': DBMSCatalog.objects.get(
                        type=DBMSType.POSTGRES, version='9.6'),
                    'algorithm': AlgorithmType.GPR,
                    'target_objective': target_objectives.default()
                })
            form = SessionForm(**form_kwargs)
        context = {
            'project': project,
            'session': session,
            'form': form,
        }
        return render(request, 'edit_session.html', context)


@login_required(login_url=reverse_lazy('login'))
def edit_knobs(request, project_id, session_id):
    project = get_object_or_404(Project, pk=project_id, user=request.user)
    session = get_object_or_404(Session, pk=session_id, user=request.user)
    if request.method == 'POST':
        form = SessionKnobForm(request.POST)
        if not form.is_valid():
            return render(request, 'edit_knobs.html',
                          {'project': project, 'session': session, 'form': form})
        instance = form.instance
        instance.session = session
        instance.knob = KnobCatalog.objects.get(dbms=session.dbms,
                                                name=form.cleaned_data["name"])
        SessionKnob.objects.filter(session=instance.session, knob=instance.knob).delete()
        instance.save()
        return HttpResponse(status=204)
    else:
        knobs = SessionKnob.objects.filter(session=session).prefetch_related(
            'knob').order_by('-tunable', 'knob__name')
        forms = []
        for knob in knobs:
            knob_values = model_to_dict(knob)
            knob_values['session'] = session
            knob_values['name'] = knob.knob.name
            forms.append(SessionKnobForm(initial=knob_values))
        context = {
            'project': project,
            'session': session,
            'forms': forms
        }
        return render(request, 'edit_knobs.html', context)


@login_required(login_url=reverse_lazy('login'))
def delete_session(request, project_id):
    sids = request.POST.getlist('sessions', [])
    Session.objects.filter(pk__in=sids, user=request.user).delete()
    return redirect(reverse(
        'project_sessions',
        kwargs={'project_id': project_id}))


@login_required(login_url=reverse_lazy('login'))
def result_view(request, project_id, session_id, result_id):
    target = get_object_or_404(Result, pk=result_id)
    session = target.session

    # default_metrics = [session.target_objective]
    metric_meta = target_objectives.get_metric_metadata(session.dbms.pk, session.target_objective)
    # metric_data = JSONUtil.loads(target.metric_data.data)

    # default_metrics = {mname: metric_data[mname] * metric_meta[mname].scale
    #                    for mname in default_metrics}
    if session.tuning_session == 'no_tuning_session':
        status = None
        next_conf = ''
        next_conf_available = False
    else:
        task_tuple = JSONUtil.loads(target.task_ids)
        # For now we ignore the first subtask (i.e., preprocessing) status in GPR/DNN.
        task_ids = TaskUtil.get_task_ids_from_tuple(task_tuple)[-3:]
        tasks = TaskUtil.get_tasks(task_ids)
        status, _ = TaskUtil.get_task_status(tasks, len(task_ids))

        if status == 'SUCCESS':  # pylint: disable=simplifiable-if-statement
            next_conf_available = True
        else:
            next_conf_available = False
        next_conf = ''
        cfg = target.next_configuration
        LOG.debug("status: %s, next_conf_available: %s, next_conf: %s, type: %s",
                  status, next_conf_available, cfg, type(cfg))

    if next_conf_available:
        try:
            cfg = JSONUtil.loads(cfg)['recommendation']
            kwidth = max(len(k) for k in cfg.keys())
            vwidth = max(len(str(v)) for v in cfg.values())
            next_conf = ''
            for k, v in cfg.items():
                next_conf += '{: <{kwidth}}  = {: <{vwidth}}\n'.format(
                    k, v, kwidth=kwidth, vwidth=vwidth)
        except Exception as e:  # pylint: disable=broad-except
            LOG.exception("Failed to format the next config (type=%s): %s.\n\n%s\n",
                          type(cfg), cfg, e)

    form_labels = Result.get_labels()
    form_labels.update(LabelUtil.style_labels({
        'status': 'status',
        'next_conf': 'next configuration',
    }))
    form_labels['title'] = 'Result Info'
    context = {
        'result': target,
        'metric_meta': metric_meta,
        'status': status,
        'next_conf_available': next_conf_available,
        'next_conf': next_conf,
        'labels': form_labels,
        'project_id': project_id,
        'session_id': session_id
    }
    return render(request, 'result.html', context)


@csrf_exempt
def new_result(request):
    if request.method == 'POST':
        form = NewResultForm(request.POST, request.FILES)

        if not form.is_valid():
            LOG.warning("New result form is not valid: %s", str(form.errors))
            return HttpResponse("New result form is not valid: " + str(form.errors), status=400)
        upload_code = form.cleaned_data['upload_code']
        try:
            session = Session.objects.get(upload_code=upload_code)
        except Session.DoesNotExist:
            LOG.warning("Invalid upload code: %s", upload_code)
            return HttpResponse("Invalid upload code: " + upload_code, status=400)

        execution_times = form.cleaned_data['execution_times']
        return handle_result_files(session, request.FILES, execution_times)
    LOG.warning("Request type was not POST")
    return HttpResponse("Request type was not POST", status=400)


def handle_result_files(session, files, execution_times=None):
    # Combine into contiguous files
    files = {k: b''.join(v.chunks()).decode() for k, v in list(files.items())}

    # Load the contents of the controller's summary file
    summary = JSONUtil.loads(files['summary'])

    dbms_id = session.dbms.pk
    udm_before = {}
    udm_after = {}
    udm_all = {}
    if 'user_defined_metrics' in files:
        udm_all = JSONUtil.loads(files['user_defined_metrics'])
    target_name = session.target_objective
    target_instance = target_objectives.get_instance(dbms_id, target_name)
    if target_instance.is_udf() and len(udm_all) == 0:
        return HttpResponse('ERROR: user defined target objective {} is not uploaded!'.format(
            target_name))
    if len(udm_all) > 0:
        # Note: Here we assume that for sessions with same dbms, user defined metrics are same.
        # Otherwise there may exist inconsistency, it becomes worse after restarting web server.
        if target_instance.is_udf() and (target_name not in udm_all.keys()):
            return HttpResponse('ERROR: user defined target objective {} is not uploaded!'.format(
                target_name))
        if not target_objectives.udm_registered(dbms_id):
            target_objectives.register_udm(dbms_id, udm_all)
        for name, info in udm_all.items():
            udm_name = 'udm.' + name
            udm_before[name] = 0
            udm_after[name] = info['value']
            if MetricCatalog.objects.filter(dbms=session.dbms, name=udm_name).exists():
                continue
            udm_catalog = MetricCatalog.objects.create(dbms=session.dbms,
                                                       name=udm_name,
                                                       vartype=info['type'],
                                                       scope='global',
                                                       metric_type=MetricType.STATISTICS)
            udm_catalog.summary = 'user defined metric, not target objective'
            udm_catalog.save()
    # Find worst throughput
    past_metrics = MetricData.objects.filter(session=session)
    metric_meta = target_objectives.get_instance(session.dbms.pk, session.target_objective)
    if len(past_metrics) > 0:
        worst_metric = past_metrics.order_by('-id').first()
        worst_target_value = JSONUtil.loads(worst_metric.data)[session.target_objective]
        for past_metric in past_metrics:
            if '*' in past_metric.name:
                continue
            target_value = JSONUtil.loads(past_metric.data)[session.target_objective]
            if metric_meta.improvement == target_objectives.MORE_IS_BETTER:
                if '*' in worst_metric.name or target_value < worst_target_value:
                    worst_target_value = target_value
                    worst_metric = past_metric
            else:
                if '*' in worst_metric.name or target_value > worst_target_value:
                    worst_target_value = target_value
                    worst_metric = past_metric
        if '*' in worst_metric.name:
            LOG.debug("All previous results are invalid")
            penalty_target_value = worst_target_value
        else:
            LOG.debug("Worst target value so far is: %d", worst_target_value)
            penalty_factor = JSONUtil.loads(session.hyperparameters).get('PENALTY_FACTOR', 2)
            if metric_meta.improvement == target_objectives.MORE_IS_BETTER:
                penalty_target_value = worst_target_value / penalty_factor
            else:
                penalty_target_value = worst_target_value * penalty_factor

    # Update the past invalid results
    for past_metric in past_metrics:
        if '*' in past_metric.name:
            past_metric_data = JSONUtil.loads(past_metric.data)
            past_metric_data[session.target_objective] = penalty_target_value
            past_metric.data = JSONUtil.dumps(past_metric_data)
            past_metric.save()

    # If database crashed on restart, pull latest result and worst throughput so far
    if 'error' in summary and summary['error'] == "DB_RESTART_ERROR":

        LOG.debug("Error in restarting database")

        worst_result = Result.objects.filter(metric_data=worst_metric).first()
        last_result = Result.objects.filter(session=session).order_by("-id").first()

        # Copy worst data and modify
        knob_data = worst_result.knob_data
        knob_data.pk = None
        if last_result.next_configuration is not None:
            last_conf = JSONUtil.loads(last_result.next_configuration)
            if last_conf.get("recommendation", None) is not None:
                last_conf = last_conf["recommendation"]
                last_conf = parser.convert_dbms_knobs(last_result.dbms.pk, last_conf)
                all_knobs = JSONUtil.loads(knob_data.knobs)
                for knob in all_knobs.keys():
                    for tunable_knob in last_conf.keys():
                        if tunable_knob in knob:
                            all_knobs[knob] = last_conf[tunable_knob]
                knob_data.knobs = JSONUtil.dumps(all_knobs)

                data_knobs = JSONUtil.loads(knob_data.data)
                for knob in data_knobs.keys():
                    for tunable_knob in last_conf.keys():
                        if tunable_knob in knob:
                            data_knobs[knob] = last_conf[tunable_knob]
                knob_data.data = JSONUtil.dumps(data_knobs)

        knob_name_parts = last_result.knob_data.name.split('*')[0].split('#')
        knob_name_parts[-1] = str(int(knob_name_parts[-1]) + 1) + '*'
        knob_data.name = '#'.join(knob_name_parts)
        knob_data.creation_time = now()
        knob_data.save()
        knob_data = KnobData.objects.filter(session=session).order_by("-id").first()

        metric_data = worst_result.metric_data
        metric_data.pk = None
        metric_name_parts = last_result.metric_data.name.split('*')[0].split('#')
        metric_name_parts[-1] = str(int(metric_name_parts[-1]) + 1) + '*'
        metric_data.name = '#'.join(metric_name_parts)
        metric_cpy = JSONUtil.loads(metric_data.data)
        metric_cpy[session.target_objective] = penalty_target_value
        metric_data.data = JSONUtil.dumps(metric_cpy)
        metric_data.creation_time = now()
        metric_data.save()
        metric_data = MetricData.objects.filter(session=session).order_by("-id").first()

        result = worst_result
        result.pk = None
        result.knob_data = knob_data
        result.metric_data = metric_data
        result.creation_time = now()
        result.observation_start_time = now()
        result.observation_end_time = now()
        result.next_configuration = {}
        result.save()
        result = Result.objects.filter(session=session).order_by("-id").first()

        knob_diffs = {}
        metric_diffs = {}

    else:
        dbms_type = DBMSType.type(summary['database_type'])
        dbms_version = summary['database_version']
        workload_name = summary['workload_name']
        observation_time = summary['observation_time']
        start_time = datetime.fromtimestamp(
            # int(summary['start_time']), # unit: seconds
            int(float(summary['start_time']) / 1000),  # unit: ms
            timezone(TIME_ZONE))
        end_time = datetime.fromtimestamp(
            # int(summary['end_time']), # unit: seconds
            int(float(summary['end_time']) / 1000),  # unit: ms
            timezone(TIME_ZONE))

        # Check if workload name only contains alpha-numeric, underscore and hyphen
        if not re.match('^[a-zA-Z0-9_-]+$', workload_name):
            return HttpResponse('Your workload name ' + workload_name + ' contains '
                                'invalid characters! It should only contain '
                                'alpha-numeric, underscore(_) and hyphen(-)')

        try:
            # Check that we support this DBMS and version
            dbms = DBMSCatalog.objects.get(
                type=dbms_type, version=dbms_version)
        except ObjectDoesNotExist:
            try:
                dbms_version = parser.parse_version_string(dbms_type, dbms_version)
            except Exception:  # pylint: disable=broad-except
                LOG.warning('Cannot parse dbms version %s', dbms_version)
                return HttpResponse('{} v{} is not yet supported.'.format(
                    dbms_type, dbms_version))
            try:
                # Check that we support this DBMS and version
                dbms = DBMSCatalog.objects.get(
                    type=dbms_type, version=dbms_version)
            except ObjectDoesNotExist:
                return HttpResponse('{} v{} is not yet supported.'.format(
                    dbms_type, dbms_version))

        if dbms != session.dbms:
            return HttpResponse('The DBMS must match the type and version '
                                'specified when creating the session. '
                                '(expected=' + session.dbms.full_name + ') '
                                '(actual=' + dbms.full_name + ')')

        # Load, process, and store the knobs in the DBMS's configuration
        knob_dict, knob_diffs = parser.parse_dbms_knobs(
            dbms.pk, JSONUtil.loads(files['knobs']))
        knob_to_convert = KnobCatalog.objects.filter(dbms=dbms).exclude(
            vartype=VarType.STRING).exclude(vartype=VarType.TIMESTAMP)
        converted_knob_dict = parser.convert_dbms_knobs(
            dbms.pk, knob_dict, knob_to_convert)
        knob_data = KnobData.objects.create_knob_data(
            session, JSONUtil.dumps(knob_dict, pprint=True, sort=True),
            JSONUtil.dumps(converted_knob_dict, pprint=True, sort=True), dbms)

        # Load, process, and store the runtime metrics exposed by the DBMS
        metrics_before = JSONUtil.loads(files['metrics_before'])
        metrics_after = JSONUtil.loads(files['metrics_after'])
        # Add user defined metrics
        if len(udm_before) > 0:
            metrics_before['global']['udm'] = udm_before
            metrics_after['global']['udm'] = udm_after
        initial_metric_dict, initial_metric_diffs = parser.parse_dbms_metrics(
            dbms.pk, metrics_before)
        final_metric_dict, final_metric_diffs = parser.parse_dbms_metrics(
            dbms.pk, metrics_after)
        metric_dict = parser.calculate_change_in_metrics(
            dbms.pk, initial_metric_dict, final_metric_dict)
        metric_diffs = OrderedDict([
            ('metrics_before', initial_metric_diffs),
            ('metrics_after', final_metric_diffs),
        ])
        numeric_metric_dict = parser.convert_dbms_metrics(
            dbms.pk, metric_dict, observation_time, session.target_objective)
        metric_data = MetricData.objects.create_metric_data(
            session, JSONUtil.dumps(metric_dict, pprint=True, sort=True),
            JSONUtil.dumps(numeric_metric_dict, pprint=True, sort=True), dbms)

        if 'status' in summary and summary['status'] == "range_test":
            # The metric should not be used for learning because the driver did not run workload
            # We tag the metric as invalid, so later they will be set to the worst result
            metric_data.name = 'range_test_' + metric_data.name + '*'
            metric_data.save()
        if 'status' in summary and summary['status'] == "default":
            # The metric should not be used for learning because the knob value is not real
            metric_data.name = 'default_' + metric_data.name
            metric_data.save()
            # Load the values in the default result into the metric_catalog
            for name in numeric_metric_dict.keys():
                if 'time_waited_micro_fg' in name or 'total_waits_fg' in name:
                    metric = MetricCatalog.objects.get(dbms=dbms, name=name)
                    metric.default = numeric_metric_dict[name]
                    metric.save()
            # Replace the default values in 'db_time' with the new values in the metric_catalog
            normalized_db_time = target_objectives.get_instance(session.dbms.pk, 'db_time')
            if normalized_db_time is not None:
                normalized_db_time.reload_default_metrics()
            numeric_metric_dict = parser.convert_dbms_metrics(
                dbms.pk, metric_dict, observation_time, session.target_objective)
            metric_data.data = JSONUtil.dumps(numeric_metric_dict)
            metric_data.save()
        # Normalize metrics by the amount of work
        if '*' not in metric_data.name and 'transaction_counter' in numeric_metric_dict.keys():
            # Find the first valid result as the base
            for prev_metric in MetricData.objects.filter(session=session):
                if '*' in prev_metric.name:
                    continue
                first_metric_data = JSONUtil.loads(prev_metric.data)
                first_transaction_counter = first_metric_data['transaction_counter']
                transaction_counter = numeric_metric_dict['transaction_counter']
                ratio = transaction_counter / first_transaction_counter
                do_not_normalize = ['transaction_counter', 'throughput_txn_per_sec']
                for name in numeric_metric_dict.keys():
                    if not any(n in name for n in do_not_normalize):
                        numeric_metric_dict[name] = numeric_metric_dict[name] / ratio
                metric_data.data = JSONUtil.dumps(numeric_metric_dict)
                metric_data.save()
                break

        # Create a new workload if this one does not already exist
        workload = Workload.objects.create_workload(
            dbms, session.hardware, workload_name, session.project)

        # Save this result
        result = Result.objects.create_result(
            session, dbms, workload, knob_data, metric_data,
            start_time, end_time, observation_time)
        result.save()

        # Workload is now modified so backgroundTasks can make calculation
        workload.status = WorkloadStatusType.MODIFIED
        workload.save()

    other_data = {}
    if execution_times:
        other_data['execution_times.csv'] = execution_times
        try:
            batch = []
            f = StringIO(execution_times)
            reader = csv.reader(f, delimiter=',')

            for module, fn, tag, start_ts, end_ts in reader:
                start_ts = float(start_ts)
                end_ts = float(end_ts)
                exec_time = end_ts - start_ts
                start_time = datetime.fromtimestamp(int(start_ts), timezone(TIME_ZONE))
                batch.append(
                    ExecutionTime(module=module, function=fn, tag=tag, start_time=start_time,
                                  execution_time=exec_time, result=result))
            ExecutionTime.objects.bulk_create(batch)
        except Exception:  # pylint: disable=broad-except
            LOG.warning("Error parsing execution times:\n%s", execution_times, exc_info=True)

    for filename, filedata in files.items():
        if filename not in ('knobs', 'metrics_before', 'metrics_after', 'summary'):
            other_data[filename] = filedata

    # Save all original data
    backup_data = BackupData.objects.create(
        result=result, raw_knobs=files['knobs'],
        raw_initial_metrics=files['metrics_before'],
        raw_final_metrics=files['metrics_after'],
        raw_summary=files['summary'],
        knob_log=JSONUtil.dumps(knob_diffs, pprint=True),
        metric_log=JSONUtil.dumps(metric_diffs, pprint=True),
        other=JSONUtil.dumps(other_data))
    backup_data.save()

    session.project.last_update = now()
    session.last_update = now()
    session.project.save()
    session.save()

    if session.tuning_session == 'no_tuning_session':
        return HttpResponse("Result stored successfully!")

    celery_status = 'celery status is unknown'
    if CHECK_CELERY:
        celery_status = utils.check_and_run_celery()
    result_id = result.pk
    response = None
    if session.algorithm == AlgorithmType.GPR:
        subtask_list = [
            ('preprocessing', (result_id, session.algorithm)),
            ('aggregate_target_results', ()),
            ('map_workload', ()),
            ('configuration_recommendation', ()),
        ]
    elif session.algorithm == AlgorithmType.DDPG:
        subtask_list = [
            ('preprocessing', (result_id, session.algorithm)),
            ('train_ddpg', ()),
            ('configuration_recommendation_ddpg', ()),
        ]
    elif session.algorithm == AlgorithmType.DNN:
        subtask_list = [
            ('preprocessing', (result_id, session.algorithm)),
            ('aggregate_target_results', ()),
            ('map_workload', ()),
            ('configuration_recommendation', ()),
        ]

    subtasks = []
    for name, args in subtask_list:
        task_id = '{}-{}'.format(name, uuid())
        s = signature(name, args=args, options={'task_id': task_id})
        subtasks.append(s)

    response = chain(*subtasks).apply_async()
    result.task_ids = JSONUtil.dumps(response.as_tuple())
    result.save()

    return HttpResponse("Result stored successfully! Running tuner...({}, status={}) Result ID:{}"
                        .format(celery_status, response.status, result_id))


@login_required(login_url=reverse_lazy('login'))
def dbms_knobs_reference(request, dbms_name, version, knob_name):
    knob = get_object_or_404(KnobCatalog, dbms__type=DBMSType.type(dbms_name),
                             dbms__version=version, name=knob_name)
    labels = KnobCatalog.get_labels()
    list_items = OrderedDict()
    if knob.category is not None:
        list_items[labels['category']] = knob.category
    list_items[labels['scope']] = knob.scope
    list_items[labels['tunable']] = knob.tunable
    list_items[labels['vartype']] = VarType.name(knob.vartype)
    if knob.unit != KnobUnitType.OTHER:
        list_items[labels['unit']] = knob.unit
    list_items[labels['default']] = knob.default
    if knob.minval is not None:
        list_items[labels['minval']] = knob.minval
    if knob.maxval is not None:
        list_items[labels['maxval']] = knob.maxval
    if knob.enumvals is not None:
        list_items[labels['enumvals']] = knob.enumvals
    if knob.summary is not None:
        description = knob.summary
        if knob.description is not None:
            description += knob.description
        list_items[labels['summary']] = description

    context = {
        'title': knob.name,
        'dbms': knob.dbms,
        'is_used': knob.tunable,
        'used_label': 'TUNABLE',
        'list_items': list_items,
    }
    return render(request, 'dbms_reference.html', context)


@login_required(login_url=reverse_lazy('login'))
def dbms_metrics_reference(request, dbms_name, version, metric_name):
    metric = get_object_or_404(
        MetricCatalog, dbms__type=DBMSType.type(dbms_name),
        dbms__version=version, name=metric_name)
    labels = MetricCatalog.get_labels()
    list_items = OrderedDict()
    list_items[labels['scope']] = metric.scope
    list_items[labels['vartype']] = VarType.name(metric.vartype)
    list_items[labels['summary']] = metric.summary
    context = {
        'title': metric.name,
        'dbms': metric.dbms,
        'is_used': metric.metric_type == MetricType.COUNTER,
        'used_label': MetricType.name(metric.metric_type),
        'list_items': list_items,
    }
    return render(request, 'dbms_reference.html', context=context)


@login_required(login_url=reverse_lazy('login'))
def knob_data_view(request, project_id, session_id, data_id):  # pylint: disable=unused-argument
    knob_data = get_object_or_404(KnobData, pk=data_id)
    labels = KnobData.get_labels()
    labels.update(LabelUtil.style_labels({
        'featured_data': 'tunable dbms parameters',
        'all_data': 'all dbms parameters',
    }))
    labels['title'] = 'DBMS Configuration'
    context = {
        'labels': labels,
        'data_type': 'knobs'
    }
    result = Result.objects.filter(knob_data=knob_data)[0]
    session = get_object_or_404(Session, pk=session_id)
    target_obj = JSONUtil.loads(result.metric_data.data)[session.target_objective]
    return dbms_data_view(request, context, knob_data, session, target_obj)


@login_required(login_url=reverse_lazy('login'))
def metric_data_view(request, project_id, session_id, data_id):  # pylint: disable=unused-argument
    metric_data = get_object_or_404(MetricData, pk=data_id)
    labels = MetricData.get_labels()
    labels.update(LabelUtil.style_labels({
        'featured_data': 'numeric dbms metrics',
        'all_data': 'all dbms metrics',
    }))
    labels['title'] = 'DBMS Metrics'
    context = {
        'labels': labels,
        'data_type': 'metrics'
    }
    result = Result.objects.filter(metric_data=metric_data)[0]
    session = get_object_or_404(Session, pk=session_id)
    target_obj = JSONUtil.loads(result.metric_data.data)[session.target_objective]
    return dbms_data_view(request, context, metric_data, session, target_obj)


def dbms_data_view(request, context, dbms_data, session, target_obj):
    data_type = context['data_type']
    dbms_id = session.dbms.pk

    def _format_knobs(_dict):
        if data_type == 'knobs' and session.dbms.type in (DBMSType.ORACLE,):
            _knob_meta = KnobCatalog.objects.filter(
                dbms_id=dbms_id, unit=KnobUnitType.BYTES)
            _parser = parser._get(dbms_id)  # pylint: disable=protected-access
            for _meta in _knob_meta:
                if _meta.name in _dict:
                    try:
                        _v = int(_dict[_meta.name])
                        _v = _parser.format_integer(_v, _meta)
                    except (ValueError, TypeError):
                        LOG.warning("Error parsing knob %s=%s.", _meta.name,
                                    _v, exc_info=True)
                    _dict[_meta.name] = _v

    if data_type == 'knobs':
        model_class = KnobData
        featured_names = set(SessionKnob.objects.filter(
            session=session, tunable=True).values_list(
                'knob__name', flat=True))
    else:
        model_class = MetricData
        featured_names = set(MetricCatalog.objects.filter(
            dbms=session.dbms, metric_type__in=MetricType.numeric()).values_list(
                'name', flat=True))

    obj_data = getattr(dbms_data, data_type)
    all_data_dict = JSONUtil.loads(obj_data)
    _format_knobs(all_data_dict)

    featured_dict = OrderedDict([(k, v) for k, v in all_data_dict.items()
                                 if k in featured_names])
    target_inst = target_objectives.get_instance(dbms_id, session.target_objective)
    target_obj_name = target_inst.pprint
    target_fmt = "({}: {{v:.0f}}{})".format(target_obj_name, target_inst.short_unit).format
    target_obj = target_fmt(v=target_obj)

    comp_id = request.GET.get('compare', 'none')
    if comp_id != 'none':
        compare_obj = model_class.objects.get(pk=comp_id)
        comp_data = getattr(compare_obj, data_type)
        comp_dict = JSONUtil.loads(comp_data)
        _format_knobs(comp_dict)

        all_data = [(k, v, comp_dict[k]) for k, v in list(all_data_dict.items())]
        featured_data = [(k, v, comp_dict[k]) for k, v in list(featured_dict.items())]

        if data_type == 'knobs':
            met_data = Result.objects.filter(knob_data=compare_obj)[0].metric_data.data
        else:
            met_data = dbms_data.data

        cmp_target_obj = JSONUtil.loads(met_data)[session.target_objective]
        cmp_target_obj = target_fmt(v=cmp_target_obj)
    else:
        all_data = list(all_data_dict.items())
        featured_data = list(featured_dict.items())
        cmp_target_obj = ""
    peer_data = model_class.objects.filter(session=session).exclude(pk=dbms_data.pk)

    context['all_data'] = all_data
    context['featured_data'] = featured_data
    context['dbms_data'] = dbms_data
    context['compare'] = comp_id
    context['peer_data'] = peer_data
    context['target_obj'] = target_obj
    context['cmp_target_obj'] = cmp_target_obj
    return render(request, 'dbms_data.html', context)


@login_required(login_url=reverse_lazy('login'))
def workload_view(request, project_id, session_id, wkld_id):  # pylint: disable=unused-argument
    workload = get_object_or_404(Workload, pk=wkld_id)
    session = get_object_or_404(Session, pk=session_id)

    knob_confs = KnobData.objects.filter(dbms=session.dbms,
                                         session=session)
    knob_conf_map = {}
    for conf in knob_confs:
        latest_result = Result.objects.filter(
            session=session, knob_data=conf, workload=workload).order_by(
                '-observation_end_time').first()
        if not latest_result:
            continue
        knob_conf_map[conf.name] = [conf, latest_result]
    knob_conf_map = OrderedDict(sorted(list(knob_conf_map.items()), key=lambda x: x[1][0].pk))
    default_knob_confs = [c for c, _ in list(knob_conf_map.values())][:5]
    LOG.debug("default_knob_confs: %s", default_knob_confs)

    metric_meta = target_objectives.get_metric_metadata(session.dbms.pk, session.target_objective)
    default_metrics = [session.target_objective]

    labels = Workload.get_labels()
    labels['title'] = 'Workload Information'
    context = {'workload': workload,
               'knob_confs': knob_conf_map,
               'metric_meta': metric_meta,
               'knob_data': default_knob_confs,
               'default_metrics': default_metrics,
               'labels': labels,
               'session_id': session_id}
    return render(request, 'workload.html', context)


@login_required(login_url=reverse_lazy('login'))
def download_next_config(request):
    data = request.GET
    result_id = data['id']
    res = Result.objects.get(pk=result_id)
    response = HttpResponse(res.next_configuration,
                            content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename=result_' + str(result_id) + '.cnf'
    return response


@login_required(login_url=reverse_lazy('login'))
def download_debug_info(request, project_id, session_id):  # pylint: disable=unused-argument
    session = Session.objects.get(pk=session_id)
    content, filename = utils.dump_debug_info(session, pretty_print=False)
    file = ContentFile(content.getvalue())
    response = HttpResponse(file, content_type='application/x-gzip')
    response['Content-Length'] = file.size
    response['Content-Disposition'] = 'attachment; filename={}.tar.gz'.format(filename)
    return response


@login_required(login_url=reverse_lazy('login'))
def download_objectives(request, project_id, session_id):  # pylint: disable=unused-argument
    session = Session.objects.get(pk=session_id)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename={}_objectives.csv'.format(session.name)
    writer = csv.writer(response)

    objectives = target_objectives.get_all(session.dbms.pk)
    labels = ['id']
    for objective_name in objectives.keys():
        labels.append(objective_name)
    writer.writerow(labels)
    metric_files = MetricData.objects.filter(session=session)
    row_cnt = 0
    for metric_file in metric_files:
        if 'range_test' not in metric_file.name:
            metric_data = JSONUtil.loads(metric_file.data)
            row_data = [str(row_cnt)]
            for objective_name in objectives.keys():
                row_data.append(metric_data.get(objective_name, -1))
            writer.writerow(row_data)
            row_cnt += 1
    return response


@login_required(login_url=reverse_lazy('login'))
def pipeline_data_view(request, pipeline_id):
    pipeline_data = PipelineData.objects.get(pk=pipeline_id)
    task_name = PipelineTaskType.TYPE_NAMES[pipeline_data.task_type]
    data = JSONUtil.loads(pipeline_data.data)
    context = {"id": pipeline_id,
               "workload": pipeline_data.workload,
               "creation_time": pipeline_data.creation_time,
               "task_name": task_name,
               "data": data}
    return render(request, "pipeline_data.html", context)


def _tuner_status_helper(project_id, session_id, result_id):  # pylint: disable=unused-argument
    res = Result.objects.get(pk=result_id)
    task_tuple = JSONUtil.loads(res.task_ids)
    task_ids = TaskUtil.get_task_ids_from_tuple(task_tuple)[-3:]
    tasks = TaskUtil.get_tasks(task_ids)

    overall_status, num_completed = TaskUtil.get_task_status(tasks, len(task_ids))
    if overall_status in ['PENDING', 'RECEIVED', 'STARTED', 'UNAVAILABLE']:
        completion_time = 'N/A'
        total_runtime = 'N/A'
    else:
        completion_time = tasks.reverse()[0].date_done
        total_runtime = (completion_time - res.creation_time).total_seconds()
        total_runtime = '{0:.2f} seconds'.format(total_runtime)

    task_info = list(zip(TaskType.TYPE_NAMES.values(), tasks))

    context = {"id": result_id,
               "result": res,
               "overall_status": overall_status,
               "num_completed": "{} / {}".format(num_completed, 3),
               "completion_time": completion_time,
               "total_runtime": total_runtime,
               "tasks": task_info}
    return context


@login_required(login_url=reverse_lazy('login'))
def tuner_status_view(request, project_id, session_id, result_id):  # pylint: disable=unused-argument
    context = _tuner_status_helper(project_id, session_id, result_id)
    return render(request, "task_status.html", context)


# Data Format
#    error
#    metrics as a list of selected metrics
#    results
#        data for each selected metric
#            meta data for the metric
#            Result list for the metric in a folded list
@login_required(login_url=reverse_lazy('login'))
def get_workload_data(request):
    data = request.GET

    workload = get_object_or_404(Workload, pk=data['id'])
    session = get_object_or_404(Session, pk=data['session_id'])
    if session.user != request.user:
        return render(request, '404.html')

    results = Result.objects.filter(workload=workload)
    result_data = {r.pk: JSONUtil.loads(r.metric_data.data) for r in results}
    results = sorted(results, key=lambda x: int(result_data[x.pk][session.target_objective]))

    default_metrics = [session.target_objective]
    metrics = request.GET.get('met', ','.join(default_metrics)).split(',')
    metrics = [m for m in metrics if m != 'none']
    if len(metrics) == 0:
        metrics = default_metrics

    data_package = {'results': [],
                    'error': 'None',
                    'metrics': metrics}
    metric_meta = target_objectives.get_metric_metadata(session.dbms.pk, session.target_objective)
    for met in data_package['metrics']:
        met_info = metric_meta[met]
        data_package['results'].append({'data': [[]], 'tick': [],
                                        'unit': met_info.unit,
                                        'lessisbetter': met_info.improvement,
                                        'metric': met_info.pprint})

        added = set()
        knob_confs = data['conf'].split(',')
        i = len(knob_confs)
        for r in results:
            metric_data = JSONUtil.loads(r.metric_data.data)
            if r.knob_data.pk in added or str(r.knob_data.pk) not in knob_confs:
                continue
            added.add(r.knob_data.pk)
            data_val = metric_data[met] * met_info.scale
            data_package['results'][-1]['data'][0].append([
                i,
                data_val,
                r.pk,
                data_val])
            data_package['results'][-1]['tick'].append(r.knob_data.name)
            i -= 1
        data_package['results'][-1]['data'].reverse()
        data_package['results'][-1]['tick'].reverse()

    return HttpResponse(JSONUtil.dumps(data_package), content_type='application/json')


# Data Format:
#    error
#    results
#        all result data after the filters for the table
#    timelines
#        data for each benchmark & metric pair
#            meta data for the pair
#            data as a map<DBMS name, result list>
@login_required(login_url=reverse_lazy('login'))
def get_timeline_data(request):
    result_labels = Result.get_labels()
    columnnames = [
        result_labels['id'],
        result_labels['creation_time'],
        result_labels['knob_data'],
        result_labels['metric_data'],
        result_labels['workload'],
    ]
    data_package = {
        'error': 'None',
        'timelines': [],
        'knobtimelines': [],
        'columnnames': columnnames,
    }

    session = get_object_or_404(Session, pk=request.GET['session'])
    if session.user != request.user:
        return HttpResponse(JSONUtil.dumps(data_package), content_type='application/json')

    default_metrics = [session.target_objective]
    metric_meta = target_objectives.get_metric_metadata(session.dbms.pk, session.target_objective)
    for met in default_metrics:
        met_info = metric_meta[met]
        columnnames.append(met_info.pprint + ' (' + met_info.short_unit + ')')

    results_per_page = int(request.GET['nres'])

    # Get all results related to the selected session, sort by time
    results = Result.objects.filter(session=session)\
        .select_related('knob_data', 'metric_data', 'workload')
    results = sorted(results, key=lambda x: x.observation_end_time)

    display_type = request.GET['wkld']
    if display_type == 'show_none':
        workloads = []
        metrics = default_metrics
        results = []
    else:
        metrics = request.GET.get('met', ','.join(default_metrics)).split(',')
        metrics = [m for m in metrics if m != 'none']
        if len(metrics) == 0:
            metrics = default_metrics
        workloads = [display_type]
        workload_confs = [wc for wc in request.GET['spe'].strip().split(',') if wc != '']
        results = [r for r in results if str(r.workload.pk) in workload_confs]

    metric_datas = {r.pk: JSONUtil.loads(r.metric_data.data) for r in results}
    result_list = []
    for res in results:
        entry = [
            res.pk,
            res.observation_end_time.astimezone(timezone(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S"),
            res.knob_data.name,
            res.metric_data.name,
            res.workload.name]
        for met in metrics:
            entry.append(metric_datas[res.pk][met] * metric_meta[met].scale)
        entry.extend([
            '',
            res.knob_data.pk,
            res.metric_data.pk,
            res.workload.pk
        ])
        result_list.append(entry)
    data_package['results'] = result_list

    # For plotting charts
    for metric in metrics:
        met_info = metric_meta[metric]
        for wkld in workloads:
            w_r = [r for r in results if r.workload.name == wkld]
            if len(w_r) == 0:
                continue

            data = {
                'workload': wkld,
                'units': met_info.unit,
                'lessisbetter': met_info.improvement,
                'data': {},
                'baseline': "None",
                'metric': metric,
                'print_metric': met_info.pprint,
            }

            for dbms in request.GET['dbms'].split(','):
                d_r = [r for r in w_r if r.dbms.key == dbms]
                d_r = d_r[-results_per_page:]
                out = []
                for res in d_r:
                    metric_data = JSONUtil.loads(res.metric_data.data)
                    out.append([
                        res.observation_end_time.astimezone(timezone(TIME_ZONE)).
                        strftime("%m-%d-%y %H:%M"),
                        metric_data[metric] * met_info.scale,
                        "",
                        str(res.pk)
                    ])

                if len(out) > 0:
                    data['data'][dbms] = out

            data_package['timelines'].append(data)

    knobs = SessionKnob.objects.get_knobs_for_session(session)
    knob_names = [knob["name"] for knob in knobs if knob["tunable"]]
    knobs = request.GET.get('knb', ','.join(knob_names)).split(',')
    knobs = [knob for knob in knobs if knob != "none"]
    LOG.debug("Knobs plotted: %s", str(knobs))
    for knob in knobs:
        data = {
            'units': KnobUnitType.TYPE_NAMES[KnobCatalog.objects.filter(name=knob)[0].unit],
            'data': [],
            'knob': knob,
        }
        for res in results:
            knob_data = JSONUtil.loads(res.knob_data.data)
            data['data'].append([
                res.observation_end_time.astimezone(timezone(TIME_ZONE)).
                strftime("%m-%d-%y %H:%M"),
                knob_data[knob],
                "",
                str(res.pk)
            ])
        data_package['knobtimelines'].append(data)
    return HttpResponse(JSONUtil.dumps(data_package), content_type='application/json')


# get the lastest result
def give_result(request, upload_code):  # pylint: disable=unused-argument
    try:
        session = Session.objects.get(upload_code=upload_code)
    except Session.DoesNotExist:
        LOG.warning("Invalid upload code: %s", upload_code)
        return HttpResponse("Invalid upload code: " + upload_code, status=400)

    if session.tuning_session == 'no_tuning_session':
        err_msg = "Session type '{}' does not generate configurations (upload only)".format(
            session.get_tuning_session_display())
        LOG.warning(err_msg)
        return HttpResponse(err_msg, status=404)

    latest_result = Result.objects.filter(session=session).latest('creation_time')
    task_tuple = JSONUtil.loads(latest_result.task_ids)
    task_res = celery.result.result_from_tuple(task_tuple)

    task_list = []
    task = task_res
    while task is not None:
        task_list.append(task)
        task = task.parent

    group_res = celery.result.GroupResult(task_res.task_id, results=task_list)
    next_config = latest_result.next_configuration

    LOG.debug("result_id: %s, succeeded: %s, failed: %s, ready: %s, tasks_completed: %s/%s, "
              "next_config: %s\n", latest_result.pk, group_res.successful(),
              group_res.failed(), group_res.ready(), group_res.completed_count(),
              len(group_res), next_config)

    response = dict(celery_status='', result_id=latest_result.pk, message='', errors=[])

    if group_res.failed():
        errors = [t.traceback for t in task_list if t.traceback]
        if errors:
            LOG.warning('\n\n'.join(errors))
        response.update(
            celery_status='FAILURE', errors=errors,
            message='Celery failed to get the next configuration')
        status_code = 400

    elif group_res.ready():
        assert group_res.successful()
        latest_result = Result.objects.filter(session=session).latest('creation_time')
        next_config = JSONUtil.loads(latest_result.next_configuration)
        response.update(
            next_config, celery_status='SUCCESS',
            message='Celery successfully recommended the next configuration')
        status_code = 200

    else:  # One or more tasks are still waiting to execute
        celery_status = 'PENDING'
        if CHECK_CELERY:
            celery_status = utils.check_and_run_celery()
        response.update(celery_status=celery_status, message='Result not ready')
        status_code = 202

    return HttpResponse(JSONUtil.dumps(response, pprint=True), status=status_code,
                        content_type='application/json')


# get the lastest result
def get_debug_info(request, upload_code):  # pylint: disable=unused-argument
    pprint = bool(int(request.GET.get('pp', False)))
    try:
        session = Session.objects.get(upload_code=upload_code)
    except Session.DoesNotExist:
        LOG.warning("Invalid upload code: %s", upload_code)
        return HttpResponse("Invalid upload code: " + upload_code, status=400)

    content, filename = utils.dump_debug_info(session, pretty_print=pprint)
    file = ContentFile(content.getvalue())
    response = HttpResponse(file, content_type='application/x-gzip')
    response['Content-Length'] = file.size
    response['Content-Disposition'] = 'attachment; filename={}.tar.gz'.format(filename)
    return response


def train_ddpg_loops(request, session_id):  # pylint: disable=unused-argument
    session = get_object_or_404(Session, pk=session_id, user=request.user)  # pylint: disable=unused-variable
    results = Result.objects.filter(session=session_id)
    for result in results:
        train_ddpg(result.pk)
    return HttpResponse()


@csrf_exempt
def alt_get_info(request, name):  # pylint: disable=unused-argument
    # Backdoor method for getting basic info
    if name in ('website', 'logs'):
        tmpdir = os.path.realpath('.info/info_{}'.format(int(time.time())))
        os.makedirs(tmpdir, exist_ok=False)

        try:
            if name == 'website':
                filepath = os.path.join(tmpdir, 'website_dump.json.gz')
                call_command('dumpwebsite', dumpfile=filepath, compress=True)
            else:  # name == 'logs'
                base_dir = 'website_log'
                base_name = os.path.join(tmpdir, base_dir)
                shutil.copytree(LOG_DIR, base_name)
                filepath = shutil.make_archive(
                    base_name, 'gztar', tmpdir, base_dir)

            f = open(filepath, 'rb')
            try:
                cfile = File(f)
                response = HttpResponse(cfile, content_type='application/x-gzip')
                response['Content-Length'] = cfile.size
                response['Content-Disposition'] = 'attachment; filename={}'.format(
                    os.path.basename(filepath))
            finally:
                f.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        info = {}
        msg = ''
        status_code = 200

        if name == 'server':
            for k in ('engine', 'name', 'port', 'host'):
                v = connection.settings_dict.get(k.upper(), '')
                if k == 'host' and not v:
                    v = 'localhost'
                info['db_' + k] = v
            info['hostname'] = socket.gethostname()
            info['git_commit_hash'] = utils.git_hash()
            msg = "Successfully retrieved info for '{}'.".format(name)
        elif name in app_models.__dict__ and hasattr(app_models.__dict__[name], 'objects'):
            data = {k: v[0] for k, v in request.POST.lists()}
            require_exists = data.pop('require_exists', False)
            obj_str = '{}({})'.format(
                name, ','.join('{}={}'.format(*o) for o in sorted(data.items())))
            try:
                obj = app_models.__dict__[name].objects.filter(**data).first()
                if obj is None:
                    msg = "No objects found matching {}.".format(obj_str)
                    LOG.warning(msg)
                    msg = ('ERROR: ' if require_exists else 'WARNING: ') + msg
                    status_code = 400 if require_exists else 200
                else:
                    info = model_to_dict(obj)
                    msg = "Successfully retrieved info for object {}.".format(obj_str)
            except FieldError as e:
                msg = "Failed to get object {}: invalid field.\n\n{}\n\n".format(obj_str, e)
                LOG.warning(msg)
                msg = 'ERROR: ' + msg
                status_code = 400
        else:
            msg = "Invalid name for info request: '{}'.".format(name)
            LOG.warning(msg)
            msg = 'ERROR: ' + msg
            status_code = 400

        content = dict(message=msg, info=info, name=name)
        response = HttpResponse(JSONUtil.dumps(content), content_type='application/json',
                                status=status_code)

    return response


def _alt_checker(request, response, required_data=None, authenticate_user=False):
    required_data = required_data or ()
    data = {k: v[0] for k, v in request.POST.lists()}

    missing = [k for k in required_data if k not in data]
    if missing:
        err_msg = "Request is missing required data: {}".format(', '.join(missing))
        response['message'] = 'ERROR: ' + err_msg
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), content_type='application/json', status=400)

    if authenticate_user:
        user = authenticate(User, username=data['username'], password=data['password'])
        if not user:
            err_msg = "Unable to authenticate user '{}'.".format(data['username'])
            LOG.warning(err_msg)
            response.update(message='ERROR: ' + err_msg)
            return HttpResponse(JSONUtil.dumps(response), content_type='application/json',
                                status=400)
        data['user'] = user

    return data


@csrf_exempt
def alt_create_user(request):
    response = dict(created=False, message=None, user=None)
    res = _alt_checker(request, response, required_data=('username', 'password'))
    if isinstance(res, HttpResponse):
        return res

    data = res
    user, created = utils.create_user(**data)
    if created:
        msg = "Successfully created user '{}'.".format(data['username'])
        LOG.info(msg)
    else:
        msg = "User '{}' already exists.".format(data['username'])
        LOG.warning(msg)
        msg = 'WARNING: ' + msg

    response.update(user=model_to_dict(user), created=created, message=msg)
    return HttpResponse(JSONUtil.dumps(response), content_type='application/json', status=200)


@csrf_exempt
def alt_delete_user(request):
    response = dict(deleted=False, message=None, delete_info=None)
    res = _alt_checker(request, response, required_data=('username',))
    if isinstance(res, HttpResponse):
        return res

    data = res
    delete_info, deleted = utils.delete_user(**data)
    if deleted:
        msg = "Successfully deleted user '{}'.".format(data['username'])
        LOG.info(msg)
    else:
        msg = "User '{}' does not exist.".format(data['username'])
        LOG.warning(msg)
        msg = 'WARNING: ' + msg

    response.update(message=msg, deleted=deleted, delete_info=delete_info)
    return HttpResponse(JSONUtil.dumps(response), content_type='application/json', status=200)


@csrf_exempt
def alt_create_or_edit_project(request):
    response = dict(created=False, updated=False, message=None, project=None)
    res = _alt_checker(request, response, required_data=('username', 'password', 'name'),
                       authenticate_user=True)
    if isinstance(res, HttpResponse):
        return res

    data = res
    user = data.pop('user')
    data.pop('username')
    data.pop('password')
    project_name = data.pop('name')

    ts = now()
    created = False
    updated = False

    if request.path == reverse('backdoor_create_project'):
        defaults = dict(creation_time=ts, last_update=ts, **data)
        project, created = Project.objects.get_or_create(
            user=user, name=project_name, defaults=defaults)

        if created:
            msg = "Successfully created project '{}'.".format(project_name)
        else:
            msg = "Project '{}' already exists.".format(project_name)
            LOG.warning(msg)
            msg = 'WARNING: ' + msg
    else:
        project = get_object_or_404(Project, name=project_name, user=user)
        for k, v in data.items():
            setattr(project, k, v)
        project.last_update = ts
        project.save()
        msg = "Successfully updated project '{}'".format(project_name)
        updated = True

    response.update(message=msg, project=model_to_dict(project), created=created, updated=updated)
    return HttpResponse(JSONUtil.dumps(response), content_type='application/json', status=200)


@csrf_exempt
def alt_create_or_edit_session(request):
    response = dict(created=False, updated=False, message=None, session=None)
    authenticate_user = True

    if request.path == reverse('backdoor_create_session'):
        required_data = (
            'username', 'password', 'project_name', 'name', 'dbms_type', 'dbms_version')
    else:
        if 'upload_code' in request.POST:
            required_data = ()
            authenticate_user = False
        else:
            required_data = ('username', 'password', 'project_name', 'name')

    res = _alt_checker(request, response, required_data=required_data,
                       authenticate_user=authenticate_user)
    if isinstance(res, HttpResponse):
        return res

    data = res
    warnings = []

    if 'hardware' in data:
        data.pop('hardware')
        warn_msg = "Custom hardware objects are not supported."
        LOG.warning(warn_msg)
        warnings.append('WARNING: ' + warn_msg)

    created = False
    updated = False
    data.pop('username', None)
    data.pop('password', None)
    user = data.pop('user', None)
    project_name = data.pop('project_name', None)
    session_name = data.pop('name', None)
    if 'algorithm' in data:
        data['algorithm'] = AlgorithmType.type(data['algorithm'])
    session_knobs = data.pop('session_knobs', None)
    disable_others = data.pop('disable_others', False)
    hyperparams = data.pop('hyperparameters', None)
    return_ddpg_model = data.pop('return_ddpg_model', False)
    ts = now()

    if request.path == reverse('backdoor_create_session'):
        defaults = {}
        project = get_object_or_404(Project, name=project_name, user=user)
        dbms_type = DBMSType.type(data.pop('dbms_type'))
        dbms_version = data.pop('dbms_version')
        defaults['dbms'] = get_object_or_404(DBMSCatalog, type=dbms_type, version=dbms_version)
        hardware, _ = Hardware.objects.get_or_create(pk=1)
        defaults['hardware'] = hardware
        defaults['upload_code'] = data.pop('upload_code', None) or MediaUtil.upload_code_generator()
        defaults.update(creation_time=ts, last_update=ts, **data)
        if 'ddpg_actor_model' in defaults:
            defaults['ddpg_actor_model'] =\
                base64.decodebytes(defaults['ddpg_actor_model'].encode('utf8'))
            defaults['ddpg_critic_model'] =\
                base64.decodebytes(defaults['ddpg_critic_model'].encode('utf8'))
            defaults['ddpg_reply_memory'] =\
                base64.decodebytes(defaults['ddpg_replay_memory'].encode('utf8'))
            # There is a typo in the object name. After correcting that typo, remove the next line.
            defaults.pop('ddpg_replay_memory')

        session, created = Session.objects.get_or_create(user=user, project=project,
                                                         name=session_name, defaults=defaults)

        if created:
            msg = "Successfully created session '{}'.".format(session_name)
            set_default_knobs(session)
        else:
            msg = "Session '{}' already exists.".format(session_name)
            LOG.warning(msg)
            msg = 'WARNING: ' + msg
    else:
        if 'upload_code' in data:
            session = get_object_or_404(Session, upload_code=data['upload_code'])
        else:
            project = get_object_or_404(Project, name=project_name, user=user)
            session = get_object_or_404(Session, name=session_name, project=project, user=user)

        for k, v in data.items():
            setattr(session, k, v)

        # Corner case: when running LHS, when the tunable knobs and/or their ranges change
        # then we must delete the pre-generated configs since they are no longer valid.
        if session_knobs and session.tuning_session == 'lhs':
            session.lhs_samples = '[]'

        session.last_update = ts
        session.save()
        msg = "Successfully updated session '{}'.".format(session_name)
        updated = True

    if created or updated:
        if session_knobs:
            session_knobs = JSONUtil.loads(session_knobs)
            SessionKnob.objects.set_knob_min_max_tunability(
                session, session_knobs, disable_others=disable_others)

        if hyperparams:
            hyperparams = JSONUtil.loads(hyperparams)
            sess_hyperparams = JSONUtil.loads(session.hyperparameters)
            invalid = []

            for k, v in hyperparams.items():
                if k in sess_hyperparams:
                    sess_hyperparams[k] = v
                else:
                    invalid.append('{}={}'.format(k, v))
            session.save()
            if invalid:
                warn_msg = "Ignored invalid hyperparameters: {}".format(', '.join(invalid))
                LOG.warning(warn_msg)
                warnings.append("WARNING: " + warn_msg)

    session.refresh_from_db()
    res = model_to_dict(session)
    res['dbms_id'] = res['dbms']
    res['dbms'] = session.dbms.full_name
    res['hardware_id'] = res['hardware']
    res['hardware'] = model_to_dict(session.hardware)
    res['algorithm'] = AlgorithmType.name(res['algorithm'])
    if return_ddpg_model:
        if session.ddpg_actor_model is not None:
            res['ddpg_actor_model'] = base64.encodebytes(session.ddpg_actor_model).decode('utf8')
            res['ddpg_critic_model'] = base64.encodebytes(session.ddpg_critic_model).decode('utf8')
            res['ddpg_replay_memory'] = base64.encodebytes(
                session.ddpg_reply_memory).decode('utf8')
        else:
            res['ddpg_actor_model'] = None
            res['ddpg_critic_model'] = None
            res['ddpg_replay_memory'] = None
    sk = SessionKnob.objects.get_knobs_for_session(session)
    sess_knobs = {}
    for knob in sk:
        sess_knobs[knob['name']] = {x: knob[x] for x in ('minval', 'maxval', 'tunable',
                                                         'upperbound', 'lowerbound')}
    res['session_knobs'] = sess_knobs

    if warnings:
        msg = '\n\n'.join(warnings + [msg])

    response.update(message=msg, session=res, created=created, updated=updated)
    return HttpResponse(JSONUtil.dumps(response), content_type='application/json', status=200)


# integration test
@csrf_exempt
def pipeline_data_ready(request):  # pylint: disable=unused-argument
    LOG.debug("Latest pipeline run: %s", PipelineRun.objects.get_latest())
    if PipelineRun.objects.get_latest() is None:
        response = "Pipeline data ready: False"
    else:
        response = "Pipeline data ready: True"
    return HttpResponse(response)


# integration test
@csrf_exempt
def create_test_website(request):  # pylint: disable=unused-argument
    if User.objects.filter(username='ottertune_test_user').exists():
        User.objects.filter(username='ottertune_test_user').delete()
    if Hardware.objects.filter(pk=1).exists():
        test_hardware = Hardware.objects.get(pk=1)
    else:
        test_hardware = Hardware.objects.create(pk=1)

    test_user = User.objects.create_user(username='ottertune_test_user',
                                         password='ottertune_test_user')
    test_project = Project.objects.create(user=test_user, name='ottertune_test_project',
                                          creation_time=now(), last_update=now())

    # create no tuning session
    s1 = Session.objects.create(name='test_session_no_tuning', tuning_session='no_tuning_session',
                                dbms_id=1, hardware=test_hardware, project=test_project,
                                creation_time=now(), last_update=now(), user=test_user,
                                upload_code='ottertuneTestNoTuning')
    set_default_knobs(s1)
    # create gpr session
    s2 = Session.objects.create(name='test_session_gpr', tuning_session='tuning_session',
                                dbms_id=1, hardware=test_hardware, project=test_project,
                                creation_time=now(), last_update=now(), algorithm=AlgorithmType.GPR,
                                upload_code='ottertuneTestTuningGPR', user=test_user)
    set_default_knobs(s2)
    # create dnn session
    s3 = Session.objects.create(name='test_session_dnn', tuning_session='tuning_session',
                                dbms_id=1, hardware=test_hardware, project=test_project,
                                creation_time=now(), last_update=now(), algorithm=AlgorithmType.DNN,
                                upload_code='ottertuneTestTuningDNN', user=test_user)
    set_default_knobs(s3)
    # create ddpg session
    s4 = Session.objects.create(name='test_session_ddpg', tuning_session='tuning_session',
                                dbms_id=1, hardware=test_hardware, project=test_project,
                                creation_time=now(), last_update=now(), user=test_user,
                                upload_code='ottertuneTestTuningDDPG',
                                algorithm=AlgorithmType.DDPG)
    set_default_knobs(s4)
    response = HttpResponse("Success: create test website successfully")
    return response


# For tuner status UI test
@csrf_exempt
def tuner_status_test(request, upload_code):  # pylint: disable=unused-argument,too-many-return-statements
    try:
        session = Session.objects.get(upload_code=upload_code)
    except Session.DoesNotExist:
        LOG.warning("Invalid upload code: %s", upload_code)
        return HttpResponse("Invalid upload code: " + upload_code, status=400)

    result = Result.objects.filter(session=session).earliest('creation_time')
    context = _tuner_status_helper(session.project.id, session.id, result.id)
    overall_status = context['overall_status']
    num_completed, num_total = context['num_completed'].replace(' ', '').split('/')
    task_info = context['tasks']
    num_tasks = len(task_info)
    if overall_status.lower() != 'success':
        return HttpResponse("Failure: overall status {} should be success".format(
            overall_status.lower()))
    if num_completed != num_total:
        return HttpResponse("Failure: #completed tasks {} != #total tasks {}".format(
            num_completed, num_total))
    if num_tasks < 3:
        return HttpResponse("Failure: number of tasks {} should >= 3".format(num_tasks))
    for i in range(num_tasks):
        name, task = task_info[i]
        result = task.result
        if i == 0:
            expected_name = TaskType.TYPE_NAMES[TaskType.PREPROCESS]
            if name != expected_name:
                return HttpResponse("Failure: the first task {} should be {}".format(
                    name, expected_name))
        elif i == 1:
            expected_name = TaskType.TYPE_NAMES[TaskType.WORKLOAD_MAPPING]
            if name != expected_name:
                return HttpResponse("Failure: the second task {} should be {}".format(
                    name, expected_name))
            if session.tuning_session == "tuning_session":
                if session.algorithm == AlgorithmType.GPR or session.algorithm == AlgorithmType.DNN:
                    if isinstance(result, dict) is True:
                        return HttpResponse("Failure: map workload when there is only 1 workload")
        elif i == 2:
            expected_name = TaskType.TYPE_NAMES[TaskType.RECOMMENDATION]
            if name != expected_name:
                return HttpResponse("Failure: the third task {} should be {}".format(
                    name, expected_name))
            if isinstance(result, dict) is False:
                return HttpResponse("Failure: wrong result for task {}".format(name))
            if 'recommendation' not in result:
                return HttpResponse("Failure: wrong result for task {}".format(name))

    return HttpResponse("Success: task status view test passes")
