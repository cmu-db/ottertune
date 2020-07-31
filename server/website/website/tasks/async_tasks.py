#
# OtterTune - async_tasks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import random
import queue
import time
import numpy as np
import tensorflow as tf
import gpflow
from pyDOE import lhs
from scipy.stats import uniform
from pytz import timezone

from celery import shared_task, Task
from celery.utils.log import get_task_logger
from djcelery.models import TaskMeta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from django.utils.datetime_safe import datetime
from analysis.ddpg.ddpg import DDPG
from analysis.gp import GPRNP
from analysis.gp_tf import GPRGD
from analysis.nn_tf import NeuralNet
from analysis.gpr import gpr_models
from analysis.gpr import ucb
from analysis.gpr.optimize import tf_optimize
from analysis.gpr.predict import gpflow_predict
from analysis.preprocessing import Bin, DummyEncoder
from analysis.constraints import ParamConstraintHelper
from website.models import (PipelineData, PipelineRun, Result, Workload, SessionKnob,
                            MetricCatalog, ExecutionTime, KnobCatalog)
from website import db
from website.types import PipelineTaskType, AlgorithmType, VarType
from website.utils import DataUtil, JSONUtil
from website.settings import ENABLE_DUMMY_ENCODER, TIME_ZONE, VIEWS_FOR_DDPG


LOG = get_task_logger(__name__)


class BaseTask(Task):  # pylint: disable=abstract-method
    abstract = True

    def __init__(self):
        self.max_retries = 0


class IgnoreResultTask(BaseTask):  # pylint: disable=abstract-method
    abstract = True

    def on_success(self, retval, task_id, args, kwargs):
        super().on_success(retval, task_id, args, kwargs)

        # Completely delete this result because it's huge and not interesting
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = None
        task_meta.save()


class MapWorkloadTask(BaseTask):  # pylint: disable=abstract-method
    abstract = True

    def on_success(self, retval, task_id, args, kwargs):
        super().on_success(retval, task_id, args, kwargs)

        task_meta = TaskMeta.objects.get(task_id=task_id)
        new_res = None

        # Replace result with formatted result
        if args[0][0]['status'] == 'good' and args[0][0]['mapped_workload'] is not None:
            new_res = {
                'scores': sorted(args[0][0]['scores'].items()),
                'mapped_workload_id': args[0][0]['mapped_workload'],
            }

        task_meta.result = new_res
        task_meta.save()


class ConfigurationRecommendation(BaseTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(ConfigurationRecommendation, self).on_success(retval, task_id, args, kwargs)

        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = retval
        task_meta.save()


def save_execution_time(start_ts, fn, result):
    end_ts = time.time()
    exec_time = end_ts - start_ts
    start_time = datetime.fromtimestamp(int(start_ts), timezone(TIME_ZONE))
    ExecutionTime.objects.create(module="celery.async_tasks", function=fn, tag="",
                                 start_time=start_time, execution_time=exec_time, result=result)
    return exec_time


def _get_task_name(session, result_id):
    if session.tuning_session == 'lhs':
        algo_name = 'LHS'
    elif session.tuning_session == 'randomly_generate':
        algo_name = 'RANDOM'
    elif session.tuning_session == 'tuning_session':
        algo_name = AlgorithmType.short_name(session.algorithm)
    else:
        LOG.warning("Unhandled session type: %s", session.tuning_session)
        algo_name = session.tuning_session
    return '{}.{}@{}#{}'.format(session.project.name, session.name, algo_name, result_id)


def _task_result_tostring(task_result):
    if isinstance(task_result, dict):
        task_dict = type(task_result)()
        for k, v in task_result.items():
            if k.startswith('X_') or k.startswith('y_') or k == 'rowlabels':
                if isinstance(v, np.ndarray):
                    v = str(v.shape)
                elif isinstance(v, list):
                    v = len(v)
                else:
                    LOG.warning("Unhandled type: k=%s, type(v)=%s, v=%s", k, type(v), v)
                    v = str(v)
            task_dict[k] = v
        task_str = JSONUtil.dumps(task_dict, pprint=True)
    else:
        task_str = str(task_result)
    return task_str


def choose_value_in_range(num1, num2):
    if num1 > 10 * num2 or num2 > 10 * num1:
        # It is important to add 1 to avoid log(0)
        log_num1 = np.log(num1 + 1)
        log_num2 = np.log(num2 + 1)
        mean = np.exp((log_num1 + log_num2) / 2)
    else:
        mean = (num1 + num2) / 2
    return mean


def calc_next_knob_range(algorithm, knob_info, newest_result, good_val, bad_val, mode):
    session = newest_result.session
    knob = KnobCatalog.objects.get(name=knob_info['name'], dbms=session.dbms)
    knob_file = newest_result.knob_data
    knob_values = JSONUtil.loads(knob_file.data)
    last_value = float(knob_values[knob.name])
    session_knob = SessionKnob.objects.get(session=session, knob=knob)
    # The collected knob value may be different from the expected value
    # We use the expected value to set the knob range
    expected_value = choose_value_in_range(good_val, bad_val)

    session_results = Result.objects.filter(session=session).order_by("-id")
    last_conf_value = ''
    if len(session_results) > 1:
        last_conf = session_results[1].next_configuration
        if last_conf is not None:
            last_conf = JSONUtil.loads(last_conf)["recommendation"]
            # The names cannot be matched directly because of the 'global.' prefix
            for name in last_conf.keys():
                if name in knob.name:
                    last_conf_value = last_conf[name]
    fomatted_expect_value = db.parser.format_dbms_knobs(
        session.dbms.pk, {knob.name: expected_value})[knob.name]

    # The last result was testing the of this knob
    if last_conf_value == fomatted_expect_value:
        # Fixme: '*' is a special symbol indicating that the knob setting is invalid
        # In the future we can add a field to indicate if the knob setting is invalid
        if '*' in knob_file.name:
            if mode == 'lowerbound':
                session_knob.lowerbound = str(int(expected_value))
            else:
                session_knob.upperbound = str(int(expected_value))
            next_value = choose_value_in_range(expected_value, good_val)
        else:
            if mode == 'lowerbound':
                session_knob.minval = str(int(expected_value))
            else:
                session_knob.maxval = str(int(expected_value))
            next_value = choose_value_in_range(expected_value, bad_val)
        session_knob.save()
    else:
        next_value = expected_value

    if mode == 'lowerbound':
        next_config = {knob.name: next_value}
    else:
        next_config = {knob.name: next_value}

    target_data = {}
    target_data['newest_result_id'] = newest_result.pk
    target_data['status'] = 'range_test'
    target_data['config_recommend'] = next_config
    LOG.debug('%s: Generated a config to test %s of %s.\n\ndata=%s\n',
              _get_task_name(session, newest_result.pk), mode, knob.name,
              _task_result_tostring(target_data))
    return True, target_data


@shared_task(base=IgnoreResultTask, name='preprocessing')
def preprocessing(result_id, algorithm):
    start_ts = time.time()
    target_data = {}
    target_data['newest_result_id'] = result_id
    newest_result = Result.objects.get(pk=result_id)
    session = newest_result.session
    knobs = SessionKnob.objects.get_knobs_for_session(session)
    task_name = _get_task_name(session, result_id)
    LOG.info("%s: Preprocessing data...", task_name)

    # Check that the minvals of tunable knobs are all decided
    for knob_info in knobs:
        if knob_info.get('lowerbound', None) is not None:
            lowerbound = float(knob_info['lowerbound'])
            minval = float(knob_info['minval'])
            if lowerbound < minval * 0.7 and minval > 10:
                # We need to do binary search to determine the minval of this knob
                successful, target_data = calc_next_knob_range(
                    algorithm, knob_info, newest_result, minval, lowerbound, 'lowerbound')
                if successful:
                    exec_time = save_execution_time(start_ts, "preprocessing", newest_result)
                    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data))
                    LOG.info("%s, Done processing. Returning config for lowerbound knob search"
                             " (%.1f seconds).", task_name, exec_time)
                    return result_id, algorithm, target_data

    # Check that the maxvals of tunable knobs are all decided
    for knob_info in knobs:
        if knob_info.get('upperbound', None) is not None:
            upperbound = float(knob_info['upperbound'])
            maxval = float(knob_info['maxval'])
            if upperbound > maxval * 1.3:
                # We need to do binary search to determine the maxval of this knob
                successful, target_data = calc_next_knob_range(
                    algorithm, knob_info, newest_result, maxval, upperbound, 'upperbound')
                if successful:
                    exec_time = save_execution_time(start_ts, "preprocessing", newest_result)
                    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data))
                    LOG.info("%s: Done processing. Returning config for upperbound knob search"
                             " (%.1f seconds).", task_name, exec_time)
                    return result_id, algorithm, target_data

    # Check that we've completed the background tasks at least once. We need
    # this data in order to make a configuration recommendation (until we
    # implement a sampling technique to generate new training data).
    has_pipeline_data = PipelineData.objects.filter(workload=newest_result.workload).exists()
    session_results = Result.objects.filter(session=session)
    results_cnt = len(session_results)
    skip_ddpg = False
    ignore = ['range_test']
    for i, result in enumerate(session_results):
        if any(symbol in result.metric_data.name for symbol in ignore):
            results_cnt -= 1
            if i == len(session_results) - 1 and algorithm == AlgorithmType.DDPG:
                skip_ddpg = True

    LOG.debug("%s: workload=%s, has_pipeline_data: %s, # results: %s, results_cnt: %s, "
              "skip_ddpg: %s", task_name, newest_result.workload, has_pipeline_data,
              len(session_results), results_cnt, skip_ddpg)

    if session.tuning_session == 'randomly_generate':
        # generate a config randomly
        random_knob_result = gen_random_data(knobs)
        target_data['status'] = 'random'
        target_data['config_recommend'] = random_knob_result
        LOG.debug('%s: Generated a random config.', task_name)

    elif not has_pipeline_data or results_cnt == 0 or skip_ddpg or session.tuning_session == 'lhs':
        if not has_pipeline_data and session.tuning_session == 'tuning_session':
            LOG.info("%s: Background tasks haven't ran for this workload yet, "
                     "picking data with lhs.", task_name)
            target_data['debug'] = ("Background tasks haven't ran for this workload yet. "
                                    "If this keeps happening, please make sure Celery periodic "
                                    "tasks are running on the server.")
        if results_cnt == 0 and session.tuning_session == 'tuning_session':
            LOG.info("%s: Not enough data in this session, picking data with lhs.", task_name)
            target_data['debug'] = "Not enough data in this session, picking data with lhs."
        if skip_ddpg:
            LOG.info("%s: The most recent result cannot be used by DDPG, picking data with lhs.",
                     task_name)
            target_data['debug'] = ("The most recent result cannot be used by DDPG,"
                                    "picking data with lhs.")

        all_samples = JSONUtil.loads(session.lhs_samples)
        if len(all_samples) == 0:
            num_lhs_samples = 100 if session.tuning_session == 'lhs' else 10
            all_samples = gen_lhs_samples(knobs, num_lhs_samples)
            LOG.debug('%s: Generated %s LHS samples (LHS data: %s).', task_name, num_lhs_samples,
                      len(all_samples))
        samples = all_samples.pop()
        target_data['status'] = 'lhs'
        target_data['config_recommend'] = samples
        session.lhs_samples = JSONUtil.dumps(all_samples)
        session.save()
        LOG.debug('%s: Got LHS config.', task_name)

    exec_time = save_execution_time(start_ts, "preprocessing", newest_result)
    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data))
    LOG.info("%s: Done preprocessing data (%.1f seconds).", task_name, exec_time)
    return result_id, algorithm, target_data


@shared_task(base=IgnoreResultTask, name='aggregate_target_results')
def aggregate_target_results(aggregate_target_results_input):
    start_ts = time.time()
    result_id, algorithm, target_data = aggregate_target_results_input
    newest_result = Result.objects.get(pk=result_id)
    session = newest_result.session
    task_name = _get_task_name(session, result_id)

    # If the preprocessing method has already generated a config, bypass this method.
    if 'config_recommend' in target_data:
        assert 'newest_result_id' in target_data and 'status' in target_data
        LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data))
        LOG.info('%s: Skipping aggregate_target_results (status=%s).', task_name,
                 target_data.get('status', ''))
        return target_data, algorithm

    LOG.info("%s: Aggregating target results...", task_name)

    # Aggregate all knob config results tried by the target so far in this
    # tuning session and this tuning workload.
    target_results = Result.objects.filter(dbms=newest_result.dbms,
                                           workload=newest_result.workload)
    LOG.debug("%s: # results: %s", task_name, len(target_results))
    if len(target_results) == 0:
        raise Exception('Cannot find any results for session_id={}, dbms_id={}'
                        .format(session, newest_result.dbms))
    agg_data = DataUtil.aggregate_data(target_results)
    LOG.debug("%s ~ INITIAL: X_matrix=%s, X_columnlabels=%s", task_name,
              agg_data['X_matrix'].shape, len(agg_data['X_columnlabels']))
    agg_data['newest_result_id'] = result_id
    agg_data['status'] = 'good'

    # Clean knob data
    cleaned_agg_data = DataUtil.clean_knob_data(agg_data['X_matrix'],
                                                agg_data['X_columnlabels'], [session])
    agg_data['X_matrix'] = np.array(cleaned_agg_data[0])
    agg_data['X_columnlabels'] = np.array(cleaned_agg_data[1])
    LOG.debug("%s ~ FINAL: X_matrix=%s, X_columnlabels=%s", task_name,
              agg_data['X_matrix'].shape, len(agg_data['X_columnlabels']))

    exec_time = save_execution_time(start_ts, "aggregate_target_results", newest_result)
    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(agg_data))
    LOG.info('%s: Finished aggregating target results (%.1f seconds).', task_name, exec_time)
    return agg_data, algorithm


def gen_random_data(knobs):
    random_knob_result = {}
    for knob in knobs:
        name = knob["name"]
        if knob["vartype"] == VarType.BOOL:
            flag = random.randint(0, 1)
            if flag == 0:
                random_knob_result[name] = False
            else:
                random_knob_result[name] = True
        elif knob["vartype"] == VarType.ENUM:
            enumvals = knob["enumvals"].split(',')
            enumvals_len = len(enumvals)
            rand_idx = random.randint(0, enumvals_len - 1)
            random_knob_result[name] = rand_idx
        elif knob["vartype"] == VarType.INTEGER:
            random_knob_result[name] = random.randint(int(knob["minval"]), int(knob["maxval"]))
        elif knob["vartype"] == VarType.REAL:
            random_knob_result[name] = random.uniform(
                float(knob["minval"]), float(knob["maxval"]))
        elif knob["vartype"] == VarType.STRING:
            random_knob_result[name] = "None"
        elif knob["vartype"] == VarType.TIMESTAMP:
            random_knob_result[name] = "None"
        else:
            raise Exception(
                'Unknown variable type: {}'.format(knob["vartype"]))

    return random_knob_result


def gen_lhs_samples(knobs, nsamples):
    names = []
    maxvals = []
    minvals = []
    types = []

    for knob in knobs:
        names.append(knob['name'])
        maxvals.append(float(knob['maxval']))
        minvals.append(float(knob['minval']))
        types.append(knob['vartype'])

    nfeats = len(knobs)
    samples = lhs(nfeats, samples=nsamples, criterion='maximin')
    maxvals = np.array(maxvals)
    minvals = np.array(minvals)
    scales = maxvals - minvals
    for fidx in range(nfeats):
        samples[:, fidx] = uniform(loc=minvals[fidx], scale=scales[fidx]).ppf(samples[:, fidx])
    lhs_samples = []
    for sidx in range(nsamples):
        lhs_samples.append(dict())
        for fidx in range(nfeats):
            if types[fidx] == VarType.INTEGER:
                lhs_samples[-1][names[fidx]] = int(round(samples[sidx][fidx]))
            elif types[fidx] == VarType.BOOL:
                lhs_samples[-1][names[fidx]] = int(round(samples[sidx][fidx]))
            elif types[fidx] == VarType.ENUM:
                lhs_samples[-1][names[fidx]] = int(round(samples[sidx][fidx]))
            elif types[fidx] == VarType.REAL:
                lhs_samples[-1][names[fidx]] = float(samples[sidx][fidx])
            else:
                LOG.warning("LHS: vartype not supported: %s (knob name: %s).",
                            VarType.name(types[fidx]), names[fidx])
    random.shuffle(lhs_samples)

    return lhs_samples


@shared_task(base=IgnoreResultTask, name='train_ddpg')
def train_ddpg(train_ddpg_input):
    start_ts = time.time()
    result_id, algorithm, target_data = train_ddpg_input
    result = Result.objects.get(pk=result_id)
    session = result.session
    dbms = session.dbms
    task_name = _get_task_name(session, result_id)

    # If the preprocessing method has already generated a config, bypass this method.
    if 'config_recommend' in target_data:
        assert 'newest_result_id' in target_data and 'status' in target_data
        LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data))
        LOG.info("%s: Skipping train DDPG (status=%s).", task_name, target_data['status'])
        return target_data, algorithm

    LOG.info('%s: Add training data to ddpg and train ddpg...', task_name)

    params = JSONUtil.loads(session.hyperparameters)
    session_results = Result.objects.filter(session=session,
                                            creation_time__lt=result.creation_time).order_by('pk')

    results_cnt = len(session_results)
    first_valid_result = -1
    ignore = ['range_test']
    for i, result in enumerate(session_results):
        if any(symbol in result.metric_data.name for symbol in ignore):
            results_cnt -= 1
        else:
            last_valid_result = i
            first_valid_result = i if first_valid_result == -1 else first_valid_result
    target_data = {}
    target_data['newest_result_id'] = result_id

    # Extract data from result and previous results
    result = Result.objects.filter(pk=result_id)
    if results_cnt == 0:
        base_result_id = result_id
        prev_result_id = result_id
    else:
        base_result_id = session_results[first_valid_result].pk
        prev_result_id = session_results[last_valid_result].pk
    base_result = Result.objects.filter(pk=base_result_id)
    prev_result = Result.objects.filter(pk=prev_result_id)

    agg_data = DataUtil.aggregate_data(result, ignore)
    prev_agg_data = DataUtil.aggregate_data(prev_result, ignore)
    metric_data = agg_data['y_matrix'].flatten()
    prev_metric_data = prev_agg_data['y_matrix'].flatten()
    base_metric_data = (DataUtil.aggregate_data(base_result, ignore))['y_matrix'].flatten()

    target_objective = session.target_objective
    # Filter ys by current target objective metric
    target_obj_idx = [i for i, n in enumerate(agg_data['y_columnlabels']) if n == target_objective]
    if len(target_obj_idx) == 0:
        raise Exception(('[{}] Could not find target objective in metrics '
                         '(target_obj={})').format(task_name, target_objective))
    if len(target_obj_idx) > 1:
        raise Exception(('[{}] Found {} instances of target objective in '
                         'metrics (target_obj={})').format(task_name,
                                                           len(target_obj_idx),
                                                           target_objective))
    objective = metric_data[target_obj_idx]
    base_objective = base_metric_data[target_obj_idx]
    prev_objective = prev_metric_data[target_obj_idx]
    LOG.info('Target objective value:  current: %s, base: %s, previous: %s',
             objective, base_objective, prev_objective)

    # Clean metric data
    views = VIEWS_FOR_DDPG.get(dbms.type, None)
    metric_data, _ = DataUtil.clean_metric_data(agg_data['y_matrix'],
                                                agg_data['y_columnlabels'], views)
    metric_data = metric_data.flatten()
    metric_scalar = MinMaxScaler().fit(metric_data.reshape(1, -1))
    normalized_metric_data = metric_scalar.transform(metric_data.reshape(1, -1))[0]
    prev_metric_data, _ = DataUtil.clean_metric_data(prev_agg_data['y_matrix'],
                                                     prev_agg_data['y_columnlabels'], views)
    prev_metric_data = prev_metric_data.flatten()
    prev_metric_scalar = MinMaxScaler().fit(prev_metric_data.reshape(1, -1))
    prev_normalized_metric_data = prev_metric_scalar.transform(prev_metric_data.reshape(1, -1))[0]

    # Clean knob data
    cleaned_knob_data = DataUtil.clean_knob_data(agg_data['X_matrix'],
                                                 agg_data['X_columnlabels'], [session])
    knob_data = np.array(cleaned_knob_data[0])
    knob_labels = np.array(cleaned_knob_data[1])
    knob_bounds = np.vstack(DataUtil.get_knob_bounds(knob_labels.flatten(), session))
    knob_data = MinMaxScaler().fit(knob_bounds).transform(knob_data)[0]
    knob_num = len(knob_data)
    metric_num = len(metric_data)
    LOG.debug('%s: knob_num: %d, metric_num: %d', task_name, knob_num, metric_num)

    metric_meta = db.target_objectives.get_metric_metadata(session.dbms.pk,
                                                           session.target_objective)

    # Calculate the reward
    if params['DDPG_SIMPLE_REWARD']:
        objective = objective / base_objective
        prev_normalized_metric_data = normalized_metric_data
        if metric_meta[target_objective].improvement == '(less is better)':
            reward = -objective
        else:
            reward = objective
    else:
        if metric_meta[target_objective].improvement == '(less is better)':
            if objective - base_objective <= 0:  # positive reward
                reward = (np.square((2 * base_objective - objective) / base_objective) - 1)\
                    * abs(2 * prev_objective - objective) / prev_objective
            else:  # negative reward
                reward = -(np.square(objective / base_objective) - 1) * objective / prev_objective
        else:
            if objective - base_objective > 0:  # positive reward
                reward = (np.square(objective / base_objective) - 1) * objective / prev_objective
            else:  # negative reward
                reward = -(np.square((2 * base_objective - objective) / base_objective) - 1)\
                    * abs(2 * prev_objective - objective) / prev_objective
    LOG.info('%s: reward: %f', task_name, reward)

    # Update ddpg
    ddpg = DDPG(n_actions=knob_num, n_states=metric_num, alr=params['DDPG_ACTOR_LEARNING_RATE'],
                clr=params['DDPG_CRITIC_LEARNING_RATE'], gamma=params['DDPG_GAMMA'],
                batch_size=params['DDPG_BATCH_SIZE'],
                a_hidden_sizes=params['DDPG_ACTOR_HIDDEN_SIZES'],
                c_hidden_sizes=params['DDPG_CRITIC_HIDDEN_SIZES'],
                use_default=params['DDPG_USE_DEFAULT'])
    if session.ddpg_actor_model and session.ddpg_critic_model:
        ddpg.set_model(session.ddpg_actor_model, session.ddpg_critic_model)
    if session.ddpg_reply_memory:
        ddpg.replay_memory.set(session.ddpg_reply_memory)
    ddpg.add_sample(prev_normalized_metric_data, knob_data, reward, normalized_metric_data)
    for _ in range(params['DDPG_UPDATE_EPOCHS']):
        ddpg.update()
    session.ddpg_actor_model, session.ddpg_critic_model = ddpg.get_model()
    session.ddpg_reply_memory = ddpg.replay_memory.get()
    session.save()
    exec_time = save_execution_time(start_ts, "train_ddpg", result.first())
    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data))
    LOG.info('%s: Done training ddpg (%.1f seconds).', task_name, exec_time)
    return target_data, algorithm


def create_and_save_recommendation(recommended_knobs, result, status, **kwargs):
    dbms_id = result.dbms.pk
    formatted_knobs = db.parser.format_dbms_knobs(dbms_id, recommended_knobs)
    config = db.parser.create_knob_configuration(dbms_id, formatted_knobs)
    knob_names = recommended_knobs.keys()
    knobs = KnobCatalog.objects.filter(name__in=knob_names)
    knob_contexts = {knob.clean_name: knob.context for knob in knobs}
    retval = dict(**kwargs)
    retval.update(
        status=status,
        result_id=result.pk,
        recommendation=config,
        context=knob_contexts
    )
    result.next_configuration = JSONUtil.dumps(retval)
    result.save()

    return retval


def check_early_return(target_data, algorithm):
    result_id = target_data['newest_result_id']
    newest_result = Result.objects.get(pk=result_id)
    if target_data.get('status', 'good') != 'good':  # No status or status is not 'good'
        if target_data['status'] == 'random':
            info = 'The config is generated by Random.'
        elif target_data['status'] == 'lhs':
            info = 'The config is generated by LHS.'
        elif target_data['status'] == 'range_test':
            info = 'Searching for valid knob ranges.'
        else:
            info = 'Unknown.'
        info += ' ' + target_data.get('debug', '')
        target_data_res = create_and_save_recommendation(
            recommended_knobs=target_data['config_recommend'], result=newest_result,
            status=target_data['status'], info=info, pipeline_run=None)
        LOG.debug('%s: Skipping configuration recommendation (status=%s).',
                  _get_task_name(newest_result.session, result_id), target_data_res['status'])
        return True, target_data_res
    return False, None


@shared_task(base=ConfigurationRecommendation, name='configuration_recommendation_ddpg')
def configuration_recommendation_ddpg(recommendation_ddpg_input):  # pylint: disable=invalid-name
    start_ts = time.time()
    target_data, algorithm = recommendation_ddpg_input
    result_id = target_data['newest_result_id']
    result_list = Result.objects.filter(pk=result_id)
    result = result_list.first()
    session = result.session
    dbms = session.dbms
    task_name = _get_task_name(session, result_id)

    early_return, target_data_res = check_early_return(target_data, algorithm)
    if early_return:
        LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data_res))
        LOG.info("%s: Returning early from config recommendation (DDPG).", task_name)
        return target_data_res

    LOG.info('%s: Recommendation the next configuration (DDPG)...', task_name)

    params = JSONUtil.loads(session.hyperparameters)
    agg_data = DataUtil.aggregate_data(result_list)
    views = VIEWS_FOR_DDPG.get(dbms.type, None)
    metric_data, _ = DataUtil.clean_metric_data(agg_data['y_matrix'],
                                                agg_data['y_columnlabels'], views)
    metric_data = metric_data.flatten()
    metric_scalar = MinMaxScaler().fit(metric_data.reshape(1, -1))
    normalized_metric_data = metric_scalar.transform(metric_data.reshape(1, -1))[0]
    cleaned_knob_data = DataUtil.clean_knob_data(agg_data['X_matrix'],
                                                 agg_data['X_columnlabels'], [session])
    knob_labels = np.array(cleaned_knob_data[1]).flatten()
    knob_num = len(knob_labels)
    metric_num = len(metric_data)

    ddpg = DDPG(n_actions=knob_num, n_states=metric_num,
                a_hidden_sizes=params['DDPG_ACTOR_HIDDEN_SIZES'],
                c_hidden_sizes=params['DDPG_CRITIC_HIDDEN_SIZES'],
                use_default=params['DDPG_USE_DEFAULT'])
    if session.ddpg_actor_model is not None and session.ddpg_critic_model is not None:
        ddpg.set_model(session.ddpg_actor_model, session.ddpg_critic_model)
    if session.ddpg_reply_memory is not None:
        ddpg.replay_memory.set(session.ddpg_reply_memory)
    knob_data = ddpg.choose_action(normalized_metric_data)

    knob_bounds = np.vstack(DataUtil.get_knob_bounds(knob_labels, session))
    knob_data = MinMaxScaler().fit(knob_bounds).inverse_transform(knob_data.reshape(1, -1))[0]
    conf_map = {k: knob_data[i] for i, k in enumerate(knob_labels)}

    target_data_res = create_and_save_recommendation(recommended_knobs=conf_map, result=result,
                                                     status='good', info='INFO: ddpg')
    exec_time = save_execution_time(start_ts, "configuration_recommendation_ddpg", result)
    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data_res))
    LOG.info("%s: Done recommending the next configuration (DDPG, %.1f seconds).",
             task_name, exec_time)

    return target_data_res


def process_training_data(target_data):
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    latest_pipeline_run = PipelineRun.objects.get(pk=target_data['pipeline_run'])
    session = newest_result.session
    params = JSONUtil.loads(session.hyperparameters)
    pipeline_data_knob = None
    pipeline_data_metric = None

    # Load mapped workload data
    if target_data['mapped_workload'] is not None:
        mapped_workload_id = target_data['mapped_workload'][0]
        mapped_workload = Workload.objects.get(pk=mapped_workload_id)
        workload_knob_data = PipelineData.objects.get(
            pipeline_run=latest_pipeline_run,
            workload=mapped_workload,
            task_type=PipelineTaskType.KNOB_DATA)
        workload_knob_data = JSONUtil.loads(workload_knob_data.data)
        workload_metric_data = PipelineData.objects.get(
            pipeline_run=latest_pipeline_run,
            workload=mapped_workload,
            task_type=PipelineTaskType.METRIC_DATA)
        workload_metric_data = JSONUtil.loads(workload_metric_data.data)
        cleaned_workload_knob_data = DataUtil.clean_knob_data(workload_knob_data["data"],
                                                              workload_knob_data["columnlabels"],
                                                              [newest_result.session])
        X_workload = np.array(cleaned_workload_knob_data[0])
        X_columnlabels = np.array(cleaned_workload_knob_data[1])
        y_workload = np.array(workload_metric_data['data'])
        y_columnlabels = np.array(workload_metric_data['columnlabels'])
        rowlabels_workload = np.array(workload_metric_data['rowlabels'])
    else:
        # combine the target_data with itself is actually adding nothing to the target_data
        X_workload = np.array(target_data['X_matrix'])
        X_columnlabels = np.array(target_data['X_columnlabels'])
        y_workload = np.array(target_data['y_matrix'])
        y_columnlabels = np.array(target_data['y_columnlabels'])
        rowlabels_workload = np.array(target_data['rowlabels'])

    # Target workload data
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    X_target = target_data['X_matrix']
    y_target = target_data['y_matrix']
    rowlabels_target = np.array(target_data['rowlabels'])

    if not np.array_equal(X_columnlabels, target_data['X_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical X columnlabels (sorted knob names)'),
                        X_columnlabels, target_data['X_columnlabels'])
    if not np.array_equal(y_columnlabels, target_data['y_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical y columnlabels (sorted metric names)'),
                        y_columnlabels, target_data['y_columnlabels'])

    # Filter ys by current target objective metric
    target_objective = newest_result.session.target_objective
    target_obj_idx = [i for i, cl in enumerate(y_columnlabels) if cl == target_objective]
    if len(target_obj_idx) == 0:
        raise Exception(('Could not find target objective in metrics '
                         '(target_obj={})').format(target_objective))
    elif len(target_obj_idx) > 1:
        raise Exception(('Found {} instances of target objective in '
                         'metrics (target_obj={})').format(len(target_obj_idx),
                                                           target_objective))

    y_workload = y_workload[:, target_obj_idx]
    y_target = y_target[:, target_obj_idx]
    y_columnlabels = y_columnlabels[target_obj_idx]

    # Combine duplicate rows in the target/workload data (separately)
    X_workload, y_workload, rowlabels_workload = DataUtil.combine_duplicate_rows(
        X_workload, y_workload, rowlabels_workload)
    X_target, y_target, rowlabels_target = DataUtil.combine_duplicate_rows(
        X_target, y_target, rowlabels_target)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    dups_filter = np.ones(X_workload.shape[0], dtype=bool)
    target_row_tups = [tuple(row) for row in X_target]
    for i, row in enumerate(X_workload):
        if tuple(row) in target_row_tups:
            dups_filter[i] = False
    X_workload = X_workload[dups_filter, :]
    y_workload = y_workload[dups_filter, :]
    rowlabels_workload = rowlabels_workload[dups_filter]

    # Combine target & workload Xs for preprocessing
    X_matrix = np.vstack([X_target, X_workload])

    # Dummy encode categorial variables
    if ENABLE_DUMMY_ENCODER:
        categorical_info = DataUtil.dummy_encoder_helper(X_columnlabels,
                                                         newest_result.dbms)
        dummy_encoder = DummyEncoder(categorical_info['n_values'],
                                     categorical_info['categorical_features'],
                                     categorical_info['cat_columnlabels'],
                                     categorical_info['noncat_columnlabels'])
        X_matrix = dummy_encoder.fit_transform(X_matrix)
        binary_encoder = categorical_info['binary_vars']
        # below two variables are needed for correctly determing max/min on dummies
        binary_index_set = set(categorical_info['binary_vars'])
        total_dummies = dummy_encoder.total_dummies()
    else:
        dummy_encoder = None
        binary_encoder = None
        binary_index_set = []
        total_dummies = 0

    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target.shape[0] < 5:  # FIXME
        # FIXME (dva): if there are fewer than 5 target results so far
        # then scale the y values (metrics) using the workload's
        # y_scaler. I'm not sure if 5 is the right cutoff.
        y_target_scaler = None
        y_workload_scaler = StandardScaler()
        y_matrix = np.vstack([y_target, y_workload])
        y_scaled = y_workload_scaler.fit_transform(y_matrix)
    else:
        # FIXME (dva): otherwise try to compute a separate y_scaler for
        # the target and scale them separately.
        try:
            y_target_scaler = StandardScaler()
            y_workload_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target)
            y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
            y_scaled = np.vstack([y_target_scaled, y_workload_scaled])
        except ValueError:
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_scaled = y_workload_scaler.fit_transform(y_target)

    metric_meta = db.target_objectives.get_metric_metadata(
        newest_result.session.dbms.pk, newest_result.session.target_objective)
    lessisbetter = metric_meta[target_objective].improvement == db.target_objectives.LESS_IS_BETTER
    # Maximize the throughput, moreisbetter
    # Use gradient descent to minimize -throughput
    if not lessisbetter:
        y_scaled = -y_scaled

    # Set up constraint helper
    constraint_helper = ParamConstraintHelper(scaler=X_scaler,
                                              encoder=dummy_encoder,
                                              binary_vars=binary_encoder,
                                              init_flip_prob=params['INIT_FLIP_PROB'],
                                              flip_prob_decay=params['FLIP_PROB_DECAY'])

    # FIXME (dva): check if these are good values for the ridge
    # ridge = np.empty(X_scaled.shape[0])
    # ridge[:X_target.shape[0]] = 0.01
    # ridge[X_target.shape[0]:] = 0.1

    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])
    X_scaler_matrix = np.zeros([1, X_scaled.shape[1]])

    session_knobs = SessionKnob.objects.get_knobs_for_session(newest_result.session)

    # Set min/max for knob values
    for i in range(X_scaled.shape[1]):
        if i < total_dummies or i in binary_index_set:
            col_min = 0
            col_max = 1
        else:
            col_min = X_scaled[:, i].min()
            col_max = X_scaled[:, i].max()
            for knob in session_knobs:
                if X_columnlabels[i] == knob["name"]:
                    X_scaler_matrix[0][i] = knob["minval"]
                    col_min = X_scaler.transform(X_scaler_matrix)[0][i]
                    X_scaler_matrix[0][i] = knob["maxval"]
                    col_max = X_scaler.transform(X_scaler_matrix)[0][i]
        X_min[i] = col_min
        X_max[i] = col_max

    return X_columnlabels, X_scaler, X_scaled, y_scaled, X_max, X_min,\
        dummy_encoder, constraint_helper, pipeline_data_knob, pipeline_data_metric


@shared_task(base=ConfigurationRecommendation, name='configuration_recommendation')
def configuration_recommendation(recommendation_input):
    start_ts = time.time()
    target_data, algorithm = recommendation_input
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    session = newest_result.session
    task_name = _get_task_name(session, target_data['newest_result_id'])

    early_return, target_data_res = check_early_return(target_data, algorithm)
    if early_return:
        LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(target_data_res))
        LOG.info("%s: Returning early from config recommendation.", task_name)
        return target_data_res

    LOG.info("%s: Recommending the next configuration...", task_name)
    params = JSONUtil.loads(session.hyperparameters)

    X_columnlabels, X_scaler, X_scaled, y_scaled, X_max, X_min,\
        dummy_encoder, constraint_helper, pipeline_knobs,\
        pipeline_metrics = process_training_data(target_data)

    # FIXME: we should generate more samples and use a smarter sampling technique
    num_samples = params['NUM_SAMPLES']
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    for i in range(X_scaled.shape[1]):
        X_samples[:, i] = np.random.rand(num_samples) * (X_max[i] - X_min[i]) + X_min[i]

    q = queue.PriorityQueue()
    for x in range(0, y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))

    i = 0
    while i < params['TOP_NUM_CONFIG']:
        try:
            item = q.get_nowait()
            # Tensorflow get broken if we use the training data points as
            # starting points for GPRGD. We add a small bias for the
            # starting points. GPR_EPS default value is 0.001
            # if the starting point is X_max, we minus a small bias to
            # make sure it is within the range.
            dist = sum(np.square(X_max - X_scaled[item[1]]))
            if dist < 0.001:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] - abs(params['GPR_EPS'])))
            else:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] + abs(params['GPR_EPS'])))
            i = i + 1
        except queue.Empty:
            break

    res = None
    info_msg = 'INFO: training data size is {}. '.format(X_scaled.shape[0])
    if algorithm == AlgorithmType.DNN:
        info_msg += 'Recommended by DNN.'
        # neural network model
        model_nn = NeuralNet(n_input=X_samples.shape[1],
                             batch_size=X_samples.shape[0],
                             explore_iters=params['DNN_EXPLORE_ITER'],
                             noise_scale_begin=params['DNN_NOISE_SCALE_BEGIN'],
                             noise_scale_end=params['DNN_NOISE_SCALE_END'],
                             debug=params['DNN_DEBUG'],
                             debug_interval=params['DNN_DEBUG_INTERVAL'])
        if session.dnn_model is not None:
            model_nn.set_weights_bin(session.dnn_model)
        model_nn.fit(X_scaled, y_scaled, fit_epochs=params['DNN_TRAIN_ITER'])
        res = model_nn.recommend(X_samples, X_min, X_max,
                                 explore=params['DNN_EXPLORE'],
                                 recommend_epochs=params['DNN_GD_ITER'])
        session.dnn_model = model_nn.get_weights_bin()
        session.save()

    elif algorithm == AlgorithmType.GPR:
        info_msg += 'Recommended by GPR.'
        # default gpr model
        if params['GPR_USE_GPFLOW']:
            LOG.debug("%s: Running GPR with GPFLOW.", task_name)
            model_kwargs = {}
            model_kwargs['model_learning_rate'] = params['GPR_HP_LEARNING_RATE']
            model_kwargs['model_maxiter'] = params['GPR_HP_MAX_ITER']
            opt_kwargs = {}
            opt_kwargs['learning_rate'] = params['GPR_LEARNING_RATE']
            opt_kwargs['maxiter'] = params['GPR_MAX_ITER']
            opt_kwargs['bounds'] = [X_min, X_max]
            opt_kwargs['debug'] = params['GPR_DEBUG']
            opt_kwargs['ucb_beta'] = ucb.get_ucb_beta(params['GPR_UCB_BETA'],
                                                      scale=params['GPR_UCB_SCALE'],
                                                      t=i + 1., ndim=X_scaled.shape[1])
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            gpflow.reset_default_session(graph=graph)
            m = gpr_models.create_model(params['GPR_MODEL_NAME'], X=X_scaled, y=y_scaled,
                                        **model_kwargs)
            res = tf_optimize(m.model, X_samples, **opt_kwargs)
        else:
            LOG.debug("%s: Running GPR with GPRGD.", task_name)
            model = GPRGD(length_scale=params['GPR_LENGTH_SCALE'],
                          magnitude=params['GPR_MAGNITUDE'],
                          max_train_size=params['GPR_MAX_TRAIN_SIZE'],
                          batch_size=params['GPR_BATCH_SIZE'],
                          num_threads=params['TF_NUM_THREADS'],
                          learning_rate=params['GPR_LEARNING_RATE'],
                          epsilon=params['GPR_EPSILON'],
                          max_iter=params['GPR_MAX_ITER'],
                          sigma_multiplier=params['GPR_SIGMA_MULTIPLIER'],
                          mu_multiplier=params['GPR_MU_MULTIPLIER'],
                          ridge=params['GPR_RIDGE'])
            model.fit(X_scaled, y_scaled, X_min, X_max)
            res = model.predict(X_samples, constraint_helper=constraint_helper)

    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]
    best_config = X_scaler.inverse_transform(best_config)

    if ENABLE_DUMMY_ENCODER:
        # Decode one-hot encoding into categorical knobs
        best_config = dummy_encoder.inverse_transform(best_config)

    # Although we have max/min limits in the GPRGD training session, it may
    # lose some precisions. e.g. 0.99..99 >= 1.0 may be True on the scaled data,
    # when we inversely transform the scaled data, the different becomes much larger
    # and cannot be ignored. Here we check the range on the original data
    # directly, and make sure the recommended config lies within the range
    X_min_inv = X_scaler.inverse_transform(X_min)
    X_max_inv = X_scaler.inverse_transform(X_max)
    best_config = np.minimum(best_config, X_max_inv)
    best_config = np.maximum(best_config, X_min_inv)

    conf_map = {k: best_config[i] for i, k in enumerate(X_columnlabels)}
    newest_result.pipeline_knobs = pipeline_knobs
    newest_result.pipeline_metrics = pipeline_metrics

    conf_map_res = create_and_save_recommendation(
        recommended_knobs=conf_map, result=newest_result,
        status='good', info=info_msg, pipeline_run=target_data['pipeline_run'])

    exec_time = save_execution_time(start_ts, "configuration_recommendation", newest_result)
    LOG.debug("\n%s: Result = %s\n", task_name, _task_result_tostring(conf_map_res))
    LOG.info("%s: Done recommending the next configuration (%.1f seconds).", task_name, exec_time)
    return conf_map_res


def load_data_helper(filtered_pipeline_data, workload, task_type):
    pipeline_data = filtered_pipeline_data.get(workload=workload,
                                               task_type=task_type)
    LOG.debug("PIPELINE DATA: pipeline_run=%s, workoad=%s, type=%s, data=%s...",
              pipeline_data.pipeline_run.pk, workload, PipelineTaskType.name(task_type),
              str(pipeline_data.data)[:100])
    return JSONUtil.loads(pipeline_data.data)


@shared_task(base=MapWorkloadTask, name='map_workload')
def map_workload(map_workload_input):
    start_ts = time.time()
    target_data, algorithm = map_workload_input
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    session = newest_result.session
    task_name = _get_task_name(session, target_data['newest_result_id'])

    assert target_data is not None
    if target_data['status'] != 'good':
        LOG.debug('\n%s: Result = %s\n', task_name, _task_result_tostring(target_data))
        LOG.info("%s: Skipping workload mapping (status: %s).", task_name, target_data['status'])
        return target_data, algorithm

    # Get the latest version of pipeline data that's been computed so far.
    latest_pipeline_run = PipelineRun.objects.get_latest()
    assert latest_pipeline_run is not None
    target_data['pipeline_run'] = latest_pipeline_run.pk

    LOG.info("%s: Mapping the workload...", task_name)

    params = JSONUtil.loads(session.hyperparameters)
    target_workload = newest_result.workload
    X_columnlabels = np.array(target_data['X_columnlabels'])
    y_columnlabels = np.array(target_data['y_columnlabels'])

    # Find all pipeline data belonging to the latest version with the same
    # DBMS and hardware as the target
    pipeline_data = PipelineData.objects.filter(
        pipeline_run=latest_pipeline_run,
        workload__dbms=target_workload.dbms,
        workload__hardware=target_workload.hardware,
        workload__project=target_workload.project)

    # FIXME (dva): we should also compute the global (i.e., overall)
    # pruned metrics but we just use those from the first workload for now
    initialized = False
    global_pruned_metrics = None
    pruned_metric_idxs = None

    unique_workloads = pipeline_data.values_list('workload', flat=True).distinct()

    workload_data = {}
    # Compute workload mapping data for each unique workload
    for unique_workload in unique_workloads:

        # do not include the workload of the current session
        if newest_result.workload.pk == unique_workload:
            continue
        workload_obj = Workload.objects.get(pk=unique_workload)
        wkld_results = Result.objects.filter(workload=workload_obj)
        if wkld_results.exists() is False:
            # delete the workload
            workload_obj.delete()
            continue

        # Load knob & metric data for this workload
        knob_data = load_data_helper(pipeline_data, unique_workload, PipelineTaskType.KNOB_DATA)
        knob_data["data"], knob_data["columnlabels"] =\
            DataUtil.clean_knob_data(knob_data["data"], knob_data["columnlabels"],
                                     [newest_result.session])
        metric_data = load_data_helper(pipeline_data, unique_workload, PipelineTaskType.METRIC_DATA)
        X_matrix = np.array(knob_data["data"])
        y_matrix = np.array(metric_data["data"])
        rowlabels = np.array(knob_data["rowlabels"])
        assert np.array_equal(rowlabels, metric_data["rowlabels"])

        if not initialized:
            # For now set pruned metrics to be those computed for the first workload
            global_pruned_metrics = load_data_helper(
                pipeline_data, unique_workload, PipelineTaskType.PRUNED_METRICS)
            pruned_metric_idxs = [i for i in range(y_matrix.shape[1]) if y_columnlabels[
                i] in global_pruned_metrics]

            # Filter y columnlabels by pruned_metrics
            y_columnlabels = y_columnlabels[pruned_metric_idxs]
            initialized = True

        # Filter y matrices by pruned_metrics
        y_matrix = y_matrix[:, pruned_metric_idxs]

        # Combine duplicate rows (rows with same knob settings)
        X_matrix, y_matrix, rowlabels = DataUtil.combine_duplicate_rows(
            X_matrix, y_matrix, rowlabels)

        workload_data[unique_workload] = {
            'X_matrix': X_matrix,
            'y_matrix': y_matrix,
            'rowlabels': rowlabels,
        }

    if len(workload_data) == 0:
        # The background task that aggregates the data has not finished running yet
        target_data.update(mapped_workload=None, scores=None)
        LOG.debug('%s: Result = %s\n', task_name, _task_result_tostring(target_data))
        LOG.info('%s: Skipping workload mapping because no different workload is available.',
                 task_name)
        return target_data, algorithm

    # Stack all X & y matrices for preprocessing
    Xs = np.vstack([entry['X_matrix'] for entry in list(workload_data.values())])
    ys = np.vstack([entry['y_matrix'] for entry in list(workload_data.values())])

    # Scale the X & y values, then compute the deciles for each column in y
    X_scaler = StandardScaler(copy=False)
    X_scaler.fit(Xs)
    y_scaler = StandardScaler(copy=False)
    y_scaler.fit_transform(ys)
    y_binner = Bin(bin_start=1, axis=0)
    y_binner.fit(ys)
    del Xs
    del ys

    X_target = target_data['X_matrix']
    # Filter the target's y data by the pruned metrics.
    y_target = target_data['y_matrix'][:, pruned_metric_idxs]

    # Now standardize the target's data and bin it by the deciles we just
    # calculated
    X_target = X_scaler.transform(X_target)
    y_target = y_scaler.transform(y_target)
    y_target = y_binner.transform(y_target)

    scores = {}
    for workload_id, workload_entry in list(workload_data.items()):
        predictions = np.empty_like(y_target)
        X_workload = workload_entry['X_matrix']
        X_scaled = X_scaler.transform(X_workload)
        y_workload = workload_entry['y_matrix']
        y_scaled = y_scaler.transform(y_workload)
        for j, y_col in enumerate(y_scaled.T):
            # Using this workload's data, train a Gaussian process model
            # and then predict the performance of each metric for each of
            # the knob configurations attempted so far by the target.
            y_col = y_col.reshape(-1, 1)
            if params['GPR_USE_GPFLOW']:
                model_kwargs = {'lengthscales': params['GPR_LENGTH_SCALE'],
                                'variance': params['GPR_MAGNITUDE'],
                                'noise_variance': params['GPR_RIDGE']}
                tf.reset_default_graph()
                graph = tf.get_default_graph()
                gpflow.reset_default_session(graph=graph)
                m = gpr_models.create_model(params['GPR_MODEL_NAME'], X=X_scaled, y=y_col,
                                            **model_kwargs)
                gpr_result = gpflow_predict(m.model, X_target)
            else:
                model = GPRNP(length_scale=params['GPR_LENGTH_SCALE'],
                              magnitude=params['GPR_MAGNITUDE'],
                              max_train_size=params['GPR_MAX_TRAIN_SIZE'],
                              batch_size=params['GPR_BATCH_SIZE'])
                model.fit(X_scaled, y_col, ridge=params['GPR_RIDGE'])
                gpr_result = model.predict(X_target)
            predictions[:, j] = gpr_result.ypreds.ravel()
        # Bin each of the predicted metric columns by deciles and then
        # compute the score (i.e., distance) between the target workload
        # and each of the known workloads
        predictions = y_binner.transform(predictions)
        dists = np.sqrt(np.sum(np.square(
            np.subtract(predictions, y_target)), axis=1))
        scores[workload_id] = np.mean(dists)

    # Find the best (minimum) score
    best_score = np.inf
    best_workload_id = None
    best_workload_name = None
    scores_info = {}
    for workload_id, similarity_score in list(scores.items()):
        workload_name = Workload.objects.get(pk=workload_id).name
        if similarity_score < best_score:
            best_score = similarity_score
            best_workload_id = workload_id
            best_workload_name = workload_name
        scores_info[workload_id] = (workload_name, similarity_score)
    target_data.update(mapped_workload=(best_workload_id, best_workload_name, best_score),
                       scores=scores_info)
    exec_time = save_execution_time(start_ts, "map_workload", newest_result)
    LOG.debug('\n%s: Result = %s\n', task_name, _task_result_tostring(target_data))
    LOG.info('%s: Done mapping the workload (%.1f seconds).', task_name, exec_time)

    return target_data, algorithm
