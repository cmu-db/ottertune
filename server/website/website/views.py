#
# OtterTune - views.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# pylint: disable=too-many-lines
import logging
import datetime
import re
from collections import OrderedDict

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.forms import PasswordChangeForm
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile
from django.db.utils import IntegrityError
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

from . import utils
from .db import parser, target_objectives
from .forms import NewResultForm, ProjectForm, SessionForm, SessionKnobForm
from .models import (BackupData, DBMSCatalog, KnobCatalog, KnobData, MetricCatalog, User, Hardware,
                     MetricData, Project, Result, Session, Workload, SessionKnob, PipelineRun)
from .tasks import (aggregate_target_results, map_workload, train_ddpg,
                    configuration_recommendation, configuration_recommendation_ddpg)
from .types import (DBMSType, KnobUnitType, MetricType,
                    TaskType, VarType, WorkloadStatusType, AlgorithmType)
from .utils import JSONUtil, LabelUtil, MediaUtil, TaskUtil, ConversionUtil
from .settings import TIME_ZONE

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
                    'target_objective': target_objectives.default(),
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
        # knobs = KnobCatalog.objects.filter(dbms=session.dbms).order_by('-tunable')
        knobs = SessionKnob.objects.filter(session=session).order_by('-tunable', 'knob__name')
        forms = []
        for knob in knobs:
            knob_values = model_to_dict(knob)
            knob_values['session'] = session
            knob_values['name'] = KnobCatalog.objects.get(pk=knob.knob.pk).name
            # if SessionKnob.objects.filter(session=session, knob=knob).exists():
            #     new_knob = SessionKnob.objects.filter(session=session, knob=knob)[0]
            #     knob_values["minval"] = new_knob.minval
            #     knob_values["maxval"] = new_knob.maxval
            #     knob_values["tunable"] = new_knob.tunable
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

    status = None
    if target.task_ids is not None:
        tasks = TaskUtil.get_tasks(target.task_ids)
        status, _ = TaskUtil.get_task_status(tasks)
        if status is None:
            status = 'UNAVAILABLE'

    next_conf_available = True if status == 'SUCCESS' else False
    form_labels = Result.get_labels()
    form_labels.update(LabelUtil.style_labels({
        'status': 'status',
        'next_conf_available': 'next configuration'
    }))
    form_labels['title'] = 'Result Info'
    context = {
        'result': target,
        'metric_meta': metric_meta,
        'status': status,
        'next_conf_available': next_conf_available,
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

        return handle_result_files(session, request.FILES)
    LOG.warning("Request type was not POST")
    return HttpResponse("Request type was not POST", status=400)


def handle_result_files(session, files):
    from celery import chain
    # Combine into contiguous files
    files = {k: b''.join(v.chunks()).decode() for k, v in list(files.items())}

    # Load the contents of the controller's summary file
    summary = JSONUtil.loads(files['summary'])

    # If database crashed on restart, pull latest result and worst throughput so far
    if 'error' in summary and summary['error'] == "DB_RESTART_ERROR":

        LOG.debug("Error in restarting database")
        # Find worst throughput
        past_configs = MetricData.objects.filter(session=session)
        worst_throughput = None
        for curr_config in past_configs:
            throughput = JSONUtil.loads(curr_config.data)[session.target_objective]
            metric_meta = target_objectives.get_instance(
                session.dbms.pk, session.target_objective)
            if metric_meta.improvement == target_objectives.MORE_IS_BETTER:
                if worst_throughput is None or throughput < worst_throughput:
                    worst_throughput = throughput
            else:
                if worst_throughput is None or throughput > worst_throughput:
                    worst_throughput = throughput
        LOG.debug("Worst throughput so far is:%d", worst_throughput)

        result = Result.objects.filter(session=session).order_by("-id").first()
        backup_data = BackupData.objects.filter(result=result).first()
        last_conf = JSONUtil.loads(result.next_configuration)
        last_conf = last_conf["recommendation"]

        # Copy latest data and modify
        knob_data = result.knob_data
        knob_data.pk = None
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
                    unit = KnobCatalog.objects.get(dbms=session.dbms, name=knob).unit
                    bytes_system = ConversionUtil.DEFAULT_BYTES_SYSTEM
                    time_system = ConversionUtil.DEFAULT_TIME_SYSTEM
                    if unit == 1:
                        data_knobs[knob] = ConversionUtil.get_raw_size(last_conf[tunable_knob],
                                                                       bytes_system)
                    elif unit == 2:
                        data_knobs[knob] = ConversionUtil.get_raw_size(last_conf[tunable_knob],
                                                                       time_system)
                    else:
                        data_knobs[knob] = last_conf[tunable_knob]

        knob_data.data = JSONUtil.dumps(data_knobs)
        knob_data.name = knob_data.name + '*'
        knob_data.creation_time = now()
        knob_data.save()
        knob_data = KnobData.objects.filter(session=session).order_by("-id").first()

        metric_data = result.metric_data
        metric_cpy = JSONUtil.loads(metric_data.data)
        metric_cpy["throughput_txn_per_sec"] = worst_throughput
        metric_cpy = JSONUtil.dumps(metric_cpy)
        metric_data.pk = None
        metric_data.name = metric_data.name + '*'
        metric_data.data = metric_cpy
        metric_data.creation_time = now()
        metric_data.save()
        metric_data = MetricData.objects.filter(session=session).order_by("-id").first()

        result.pk = None
        result.knob_data = knob_data
        result.metric_data = metric_data
        result.creation_time = now()
        result.observation_start_time = now()
        result.observation_end_time = now()
        result.save()
        result = Result.objects.filter(session=session).order_by("-id").first()

        backup_data.pk = None
        backup_data.result = result
        backup_data.creation_time = now()
        backup_data.save()

    else:
        dbms_type = DBMSType.type(summary['database_type'])
        dbms_version = summary['database_version']  # TODO: fix parse_version_string
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
        tunable_knob_dict = parser.convert_dbms_knobs(
            dbms.pk, knob_dict)
        knob_data = KnobData.objects.create_knob_data(
            session, JSONUtil.dumps(knob_dict, pprint=True, sort=True),
            JSONUtil.dumps(tunable_knob_dict, pprint=True, sort=True), dbms)

        # Load, process, and store the runtime metrics exposed by the DBMS
        initial_metric_dict, initial_metric_diffs = parser.parse_dbms_metrics(
            dbms.pk, JSONUtil.loads(files['metrics_before']))
        final_metric_dict, final_metric_diffs = parser.parse_dbms_metrics(
            dbms.pk, JSONUtil.loads(files['metrics_after']))
        metric_dict = parser.calculate_change_in_metrics(
            dbms.pk, initial_metric_dict, final_metric_dict)
        initial_metric_diffs.extend(final_metric_diffs)
        numeric_metric_dict = parser.convert_dbms_metrics(
            dbms.pk, metric_dict, observation_time, session.target_objective)
        metric_data = MetricData.objects.create_metric_data(
            session, JSONUtil.dumps(metric_dict, pprint=True, sort=True),
            JSONUtil.dumps(numeric_metric_dict, pprint=True, sort=True), dbms)

        # Create a new workload if this one does not already exist
        workload = Workload.objects.create_workload(
            dbms, session.hardware, workload_name)

        # Save this result
        result = Result.objects.create_result(
            session, dbms, workload, knob_data, metric_data,
            start_time, end_time, observation_time)
        result.save()

        # Workload is now modified so backgroundTasks can make calculationw
        workload.status = WorkloadStatusType.MODIFIED
        workload.save()

        # Save all original data
        backup_data = BackupData.objects.create(
            result=result, raw_knobs=files['knobs'],
            raw_initial_metrics=files['metrics_before'],
            raw_final_metrics=files['metrics_after'],
            raw_summary=files['summary'],
            knob_log=knob_diffs,
            metric_log=initial_metric_diffs)
        backup_data.save()

    session.project.last_update = now()
    session.last_update = now()
    session.project.save()
    session.save()

    if session.tuning_session == 'no_tuning_session':
        return HttpResponse("Result stored successfully!")

    result_id = result.pk
    response = None
    if session.algorithm == AlgorithmType.GPR:
        response = chain(aggregate_target_results.s(result.pk, session.algorithm),
                         map_workload.s(),
                         configuration_recommendation.s()).apply_async()
    elif session.algorithm == AlgorithmType.DDPG:
        response = chain(train_ddpg.s(result.pk),
                         configuration_recommendation_ddpg.s()).apply_async()
    elif session.algorithm == AlgorithmType.DNN:
        response = chain(aggregate_target_results.s(result.pk, session.algorithm),
                         map_workload.s(),
                         configuration_recommendation.s()).apply_async()

    taskmeta_ids = []
    current_task = response
    while current_task:
        taskmeta_ids.insert(0, current_task.id)
        current_task = current_task.parent

    result.task_ids = ','.join(taskmeta_ids)
    result.save()
    return HttpResponse("Result stored successfully! Running tuner...(status={})  Result ID:{} "
                        .format(response.status, result_id))


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
    return dbms_data_view(request, context, knob_data)


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
    return dbms_data_view(request, context, metric_data)


def dbms_data_view(request, context, dbms_data):
    if context['data_type'] == 'knobs':
        model_class = KnobData
        filter_fn = parser.filter_tunable_knobs
        obj_data = dbms_data.knobs
    else:
        model_class = MetricData
        filter_fn = parser.filter_numeric_metrics
        obj_data = dbms_data.metrics

    dbms_id = dbms_data.dbms.pk
    all_data_dict = JSONUtil.loads(obj_data)
    featured_dict = filter_fn(dbms_id, all_data_dict)

    if 'compare' in request.GET and request.GET['compare'] != 'none':
        comp_id = request.GET['compare']
        compare_obj = model_class.objects.get(pk=comp_id)
        comp_data = compare_obj.knobs if \
            context['data_type'] == 'knobs' else compare_obj.metrics
        comp_dict = JSONUtil.loads(comp_data)
        comp_featured_dict = filter_fn(dbms_id, comp_dict)

        all_data = [(k, v, comp_dict[k]) for k, v in list(all_data_dict.items())]
        featured_data = [(k, v, comp_featured_dict[k])
                         for k, v in list(featured_dict.items())]
    else:
        comp_id = None
        all_data = list(all_data_dict.items())
        featured_data = list(featured_dict.items())
    peer_data = model_class.objects.filter(
        dbms=dbms_data.dbms, session=dbms_data.session)
    peer_data = [peer for peer in peer_data if peer.pk != dbms_data.pk]

    context['all_data'] = all_data
    context['featured_data'] = featured_data
    context['dbms_data'] = dbms_data
    context['compare'] = comp_id
    context['peer_data'] = peer_data
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
def tuner_status_view(request, project_id, session_id, result_id):  # pylint: disable=unused-argument
    res = Result.objects.get(pk=result_id)

    tasks = TaskUtil.get_tasks(res.task_ids)

    overall_status, num_completed = TaskUtil.get_task_status(tasks)
    if overall_status in ['PENDING', 'RECEIVED', 'STARTED', None]:
        completion_time = 'N/A'
        total_runtime = 'N/A'
    else:
        completion_time = tasks[-1].date_done
        total_runtime = (completion_time - res.creation_time).total_seconds()
        total_runtime = '{0:.2f} seconds'.format(total_runtime)

    task_info = [(tname, task) for tname, task in
                 zip(list(TaskType.TYPE_NAMES.values()), tasks)]

    context = {"id": result_id,
               "result": res,
               "overall_status": overall_status,
               "num_completed": "{} / {}".format(num_completed, 3),
               "completion_time": completion_time,
               "total_runtime": total_runtime,
               "tasks": task_info}

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
    results = sorted(results, key=lambda x: int(result_data[x.pk][target_objectives.THROUGHPUT]))

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

    def _failed_response(_latest_result, _tasks, _num_completed, _status, _msg):
        _msg = "{}\nSTATUS: {}\nRESULT ID: {}\n".format(_msg, _status, _latest_result)
        if tasks:
            _failed_task_idx = min(len(_tasks) - 1, _num_completed + 1)
            _failed_task = _tasks[_failed_task_idx]
            _msg += "TRACEBACK: {}".format(_failed_task.traceback)
        return HttpResponse(_msg, status=400)

    try:
        session = Session.objects.get(upload_code=upload_code)
    except Session.DoesNotExist:
        LOG.warning("Invalid upload code: %s", upload_code)
        return HttpResponse("Invalid upload code: " + upload_code, status=400)

    latest_result = Result.objects.filter(session=session).latest('creation_time')
    tasks = TaskUtil.get_tasks(latest_result.task_ids)
    overall_status, num_completed = TaskUtil.get_task_status(tasks)

    if overall_status == 'SUCCESS':
        if not latest_result.next_configuration:
            # If the task status was incomplete when we first queried latest_result
            # but succeeded before the call to TaskUtil.get_task_status() finished
            # then latest_result is stale and must be updated.
            LOG.debug("Updating stale result (pk=%s)", latest_result.pk)
            latest_result = Result.objects.get(id=latest_result.pk)

        if not latest_result.next_configuration:
            LOG.warning("Failed to get the next configuration from the latest result: %s",
                        model_to_dict(latest_result))
            overall_status = 'FAILURE'
            response = _failed_response(latest_result, tasks, num_completed, overall_status,
                                        'Failed to get the next configuration.')
        else:
            response = HttpResponse(JSONUtil.dumps(latest_result.next_configuration),
                                    content_type='application/json')

    elif overall_status in ('FAILURE', 'REVOKED', 'RETRY'):
        response = _failed_response(latest_result, tasks, num_completed, overall_status,
                                    'Celery failed to get the next configuration.')

    else:  # overall_status in ('PENDING', 'RECEIVED', 'STARTED'):
        response = HttpResponse("{}: Result not ready".format(overall_status), status=202)

    return response


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
def alt_create_user(request):
    response = dict(created=False, error=None, user=None)
    if request.method != 'POST':
        err_msg = "Request was not a post!"
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    data = {k: v[0] for k, v in request.POST.lists()}
    missing = [k for k in ('username', 'password') if k not in data]
    if missing:
        err_msg = "Request is missing required data: {}".format(', '.join(missing))
        LOG.warning(err_msg)
        response.update(error=err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    user, created = utils.create_user(**data)
    response.update(user=user, created=created)
    if created:
        LOG.info("Successfully created user '%s': %s", data['username'], user)
        status = 200
    else:
        err_msg = "ERROR: User '{}' already exists: {}".format(data['username'], user)
        response.update(error=err_msg)
        LOG.warning(err_msg)
        status = 400

    response['user'] = model_to_dict(response['user'])
    return HttpResponse(JSONUtil.dumps(response), status=status)


@csrf_exempt
def alt_delete_user(request):
    response = dict(deleted=False, error=None, delete_info=None)
    if request.method != 'POST':
        err_msg = "Request was not a post!"
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    data = {k: v[0] for k, v in request.POST.lists()}
    missing = [k for k in ('username',) if k not in data]
    if missing:
        err_msg = "Request is missing required data: {}".format(', '.join(missing))
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    delete_info, deleted = utils.delete_user(**data)
    response.update(deleted=deleted, delete_info=delete_info)
    if deleted:
        LOG.info("Successfully deleted user '%s': %s", data['username'], delete_info)
        status = 200
    else:
        err_msg = "User '{}' does not exist".format(data['username'])
        LOG.warning(err_msg)
        response.update(error=err_msg)
        status = 400

    return HttpResponse(JSONUtil.dumps(response), status=status)


@csrf_exempt
def alt_create_or_edit_project(request):
    response = dict(created=False, error=None, project=None)
    if request.method != 'POST':
        err_msg = "Request was not a post!"
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    data = {k: v[0] for k, v in request.POST.lists()}
    missing = [k for k in ('username', 'password', 'name') if k not in data]
    if missing:
        err_msg = "Request is missing required data: {}".format(', '.join(missing))
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    username = data.pop('username')
    password = data.pop('password')
    user = authenticate(User, username=username, password=password)
    if not user:
        err_msg = "ERROR: Unable to authenticate user '{}'.".format(username)
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    project_name = data.pop('name')
    ts = now()

    if request.path == reverse('backdoor_create_project'):
        try:
            project = Project.objects.create(user=user, name=project_name, last_update=ts,
                                             creation_time=ts, **data)
        except IntegrityError:
            err_msg = "ERROR: Project '{}' already exists.".format(project_name)
            project = Project.objects.get(user=user, name=project_name)
            response.update(error=err_msg, project=model_to_dict(project))
            LOG.warning(err_msg)
            return HttpResponse(JSONUtil.dumps(response), status=400)
    else:
        project = get_object_or_404(Project, name=project_name, user=user)
        for k, v in data.items():
            setattr(project, k, v)
        project.last_update = ts
        project.save()

    response.update(created=True, project=model_to_dict(project))
    return HttpResponse(JSONUtil.dumps(response))


@csrf_exempt
def alt_create_or_edit_session(request):
    response = dict(created=False, error=None, session=None)
    if request.method != 'POST':
        err_msg = "Request was not a post!"
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    data = {k: v[0] for k, v in request.POST.lists()}
    if 'hardware' in data:
        err_msg = "Custom hardware objects are not supported."
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    required_fields = ('username', 'password', 'project_name', 'name', 'dbms_type',
                       'dbms_version')
    missing = [k for k in required_fields if k not in data]
    if missing:
        err_msg = "Request is missing required data: {}".format(', '.join(missing))
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    username = data.pop('username')
    password = data.pop('password')
    user = authenticate(User, username=username, password=password)
    if not user:
        err_msg = "ERROR: Unable to authenticate user '{}'.".format(username)
        response.update(error=err_msg)
        LOG.warning(err_msg)
        return HttpResponse(JSONUtil.dumps(response), status=400)

    project = get_object_or_404(Project, name=data.pop('project_name'), user=user)
    dbms_type = DBMSType.type(data.pop('dbms_type'))
    dbms = get_object_or_404(DBMSCatalog, type=dbms_type, version=data.pop('dbms_version'))

    session_name = data.pop('name')
    if 'algorithm' in data:
        data['algorithm'] = AlgorithmType.type(data['algorithm'])
    session_knobs = data.pop('session_knobs', None)
    disable_others = data.pop('disable_others', False)
    ts = now()

    if request.path == reverse('backdoor_create_session'):
        hardware, _ = Hardware.objects.get_or_create(pk=1)
        upload_code = data.pop('upload_code', None) or MediaUtil.upload_code_generator()
        try:
            session = Session.objects.create(user=user, project=project, dbms=dbms,
                                             name=session_name, hardware=hardware,
                                             upload_code=upload_code, creation_time=ts,
                                             last_update=ts, **data)
        except IntegrityError:
            err_msg = "ERROR: Project '{}' already exists.".format(session_name)
            session = Session.objects.get(user=user, project=project, name=session_name)
            response.update(error=err_msg, project=model_to_dict(session))
            LOG.warning(err_msg)
            return HttpResponse(JSONUtil.dumps(response), status=400)

        set_default_knobs(session)
    else:
        session = get_object_or_404(Session, name=session_name, project=project, user=user)
        for k, v in data.items():
            setattr(session, k, v)
        session.last_update = ts
        session.save()

    if session_knobs:
        session_knobs = JSONUtil.loads(session_knobs)
        SessionKnob.objects.set_knob_min_max_tunability(session, session_knobs,
                                                        disable_others=disable_others)

    res = model_to_dict(session)
    res['dbms_id'] = res['dbms']
    res['dbms'] = session.dbms.full_name
    res['hardware_id'] = res['hardware']
    res['hardware'] = model_to_dict(session.hardware)
    res['algorithm'] = AlgorithmType.name(res['algorithm'])
    sk = SessionKnob.objects.get_knobs_for_session(session)
    sess_knobs = []
    for knob in sk:
        sess_knobs.append(dict(
            minval=knob['minval'],
            maxval=knob['maxval'],
            tunable=knob['tunable']))
    res['session_knobs'] = sess_knobs
    response.update(created=True, session=res)
    return HttpResponse(JSONUtil.dumps(response))


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
                                upload_code='ottertuneTestTuningDDPG', algorithm=AlgorithmType.DDPG)
    set_default_knobs(s4)
    response = HttpResponse("Success: create test website successfully")
    return response
