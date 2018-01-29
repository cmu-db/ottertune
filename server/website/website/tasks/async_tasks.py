#
# OtterTune - async_tasks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import numpy as np
from celery.task import task, Task
from celery.utils.log import get_task_logger
from djcelery.models import TaskMeta
from sklearn.preprocessing import StandardScaler

from analysis.gp_tf import GPR, GPRGD
from analysis.preprocessing import Bin
from website.models import PipelineData, PipelineRun, Result, Workload
from website.parser import Parser
from website.types import PipelineTaskType
from website.utils import DataUtil, JSONUtil

LOG = get_task_logger(__name__)


class UpdateTask(Task):  # pylint: disable=abstract-method

    def __init__(self):
        self.rate_limit = '50/m'
        self.max_retries = 3
        self.default_retry_delay = 60


class AggregateTargetResults(UpdateTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(AggregateTargetResults, self).on_success(retval, task_id, args, kwargs)

        # Completely delete this result because it's huge and not
        # interesting
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = None
        task_meta.save()


class MapWorkload(UpdateTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(MapWorkload, self).on_success(retval, task_id, args, kwargs)

        # Replace result with formatted result
        new_res = {
            'scores': sorted(args[0]['scores'].iteritems()),
            'mapped_workload_id': args[0]['mapped_workload'],
        }
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = new_res  # Only store scores
        task_meta.save()


class ConfigurationRecommendation(UpdateTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(ConfigurationRecommendation, self).on_success(retval, task_id, args, kwargs)

        result_id = args[0]['newest_result_id']
        result = Result.objects.get(pk=result_id)

        # Replace result with formatted result
        formatted_params = Parser.format_dbms_knobs(result.dbms.pk, retval)
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = formatted_params
        task_meta.save()

        # Create next configuration to try
        nondefault_params = JSONUtil.loads(
            result.session.nondefault_settings)
        config = Parser.create_knob_configuration(
            result.dbms.pk, formatted_params, nondefault_params)
        result.next_configuration = config
        result.save()


@task(base=AggregateTargetResults, name='aggregate_target_results')
def aggregate_target_results(result_id):
    # Check that we've completed the background tasks at least once. We need
    # this data in order to make a configuration recommendation (until we
    # implement a sampling technique to generate new training data).
    latest_pipeline_run = PipelineRun.objects.get_latest()
    if latest_pipeline_run is None:
        raise Exception("No previous data available. Implement me!")

    # Aggregate all knob config results tried by the target so far in this
    # tuning session.
    newest_result = Result.objects.get(pk=result_id)
    target_results = Result.objects.filter(
        session=newest_result.session, dbms=newest_result.dbms)
    if len(target_results) == 0:
        raise Exception('Cannot find any results for session_id={}, dbms_id={}'
                        .format(newest_result.session, newest_result.dbms))
    agg_data = DataUtil.aggregate_data(target_results)
    agg_data['newest_result_id'] = result_id
    return agg_data


@task(base=ConfigurationRecommendation, name='configuration_recommendation')
def configuration_recommendation(target_data):
    latest_pipeline_run = PipelineRun.objects.get_latest()
    assert latest_pipeline_run is not None
    assert target_data['scores'] is not None

    # Load mapped workload data
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

    X_workload = np.array(workload_knob_data['data'])
    X_columnlabels = np.array(workload_knob_data['columnlabels'])
    y_workload = np.array(workload_metric_data['data'])
    y_columnlabels = np.array(workload_metric_data['columnlabels'])
    rowlabels_workload = np.array(workload_metric_data['rowlabels'])

    # Target workload data
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    X_target = target_data['X_matrix']
    y_target = target_data['y_matrix']
    rowlabels_target = np.array(target_data['rowlabels'])

    if not np.array_equal(X_columnlabels, target_data['X_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical X columnlabels (sorted knob names)'))
    if not np.array_equal(y_columnlabels, target_data['y_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical y columnlabels (sorted metric names)'))

    # Filter Xs by top 10 ranked knobs
    ranked_knobs = PipelineData.objects.get(
        pipeline_run=latest_pipeline_run,
        workload=mapped_workload,
        task_type=PipelineTaskType.RANKED_KNOBS)
    ranked_knobs = JSONUtil.loads(ranked_knobs.data)[:10]  # FIXME
    ranked_knob_idxs = [i for i, cl in enumerate(X_columnlabels) if cl in ranked_knobs]
    X_workload = X_workload[:, ranked_knob_idxs]
    X_target = X_target[:, ranked_knob_idxs]
    X_columnlabels = X_columnlabels[ranked_knob_idxs]

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

    # Combine target & workload Xs then scale
    X_matrix = np.vstack([X_target, X_workload])
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
            y_matrix = np.vstack([y_target, y_workload])
            y_scaled = y_workload_scaler.fit_transform(y_matrix)

    # FIXME (dva): check if these are good values for the ridge
    ridge = np.empty(X_scaled.shape[0])
    ridge[:X_target.shape[0]] = 0.01
    ridge[X_target.shape[0]:] = 0.1

    # FIXME: we should generate more samples and use a smarter sampling
    # technique
    num_samples = 20
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    for i in range(X_scaled.shape[1]):
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        X_samples[:, i] = np.random.rand(
            num_samples) * (col_max - col_min) + col_min

    model = GPRGD()
    model.fit(X_scaled, y_scaled, ridge)
    res = model.predict(X_samples)

    # FIXME: whether we select the min/max for the best config depends
    # on the target objective
    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]
    best_config = X_scaler.inverse_transform(best_config)

    conf_map = {k: best_config[i] for i, k in enumerate(X_columnlabels)}
    return conf_map


def load_data_helper(filtered_pipeline_data, workload, task_type):
    pipeline_data = filtered_pipeline_data.get(workload=workload,
                                               task_type=task_type)
    LOG.debug("PIPELINE DATA: %s", str(pipeline_data.data))
    return JSONUtil.loads(pipeline_data.data)


@task(base=MapWorkload, name='map_workload')
def map_workload(target_data):
    # Get the latest version of pipeline data that's been computed so far.
    latest_pipeline_run = PipelineRun.objects.get_latest()
    assert latest_pipeline_run is not None

    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    target_workload = newest_result.workload
    X_columnlabels = np.array(target_data['X_columnlabels'])
    y_columnlabels = np.array(target_data['y_columnlabels'])

    # Find all pipeline data belonging to the latest version with the same
    # DBMS and hardware as the target
    pipeline_data = PipelineData.objects.filter(
        pipeline_run=latest_pipeline_run,
        workload__dbms=target_workload.dbms,
        workload__hardware=target_workload.hardware)

    # FIXME (dva): we should also compute the global (i.e., overall) ranked_knobs
    # and pruned metrics but we just use those from the first workload for now
    initialized = False
    global_ranked_knobs = None
    global_pruned_metrics = None
    ranked_knob_idxs = None
    pruned_metric_idxs = None

    # Compute workload mapping data for each unique workload
    unique_workloads = pipeline_data.values_list('workload', flat=True).distinct()
    assert len(unique_workloads) > 0
    workload_data = {}
    for unique_workload in unique_workloads:
        # Load knob & metric data for this workload
        knob_data = load_data_helper(pipeline_data, unique_workload, PipelineTaskType.KNOB_DATA)
        metric_data = load_data_helper(pipeline_data, unique_workload, PipelineTaskType.METRIC_DATA)
        X_matrix = np.array(knob_data["data"])
        y_matrix = np.array(metric_data["data"])
        rowlabels = np.array(knob_data["rowlabels"])
        assert np.array_equal(rowlabels, metric_data["rowlabels"])

        if not initialized:
            # For now set ranked knobs & pruned metrics to be those computed
            # for the first workload
            global_ranked_knobs = load_data_helper(
                pipeline_data, unique_workload, PipelineTaskType.RANKED_KNOBS)[:10]  # FIXME (dva)
            global_pruned_metrics = load_data_helper(
                pipeline_data, unique_workload, PipelineTaskType.PRUNED_METRICS)
            ranked_knob_idxs = [i for i in range(X_matrix.shape[1]) if X_columnlabels[
                i] in global_ranked_knobs]
            pruned_metric_idxs = [i for i in range(y_matrix.shape[1]) if y_columnlabels[
                i] in global_pruned_metrics]

            # Filter X & y columnlabels by top 10 ranked_knobs & pruned_metrics
            X_columnlabels = X_columnlabels[ranked_knob_idxs]
            y_columnlabels = y_columnlabels[pruned_metric_idxs]
            initialized = True

        # Filter X & y matrices by top 10 ranked_knobs & pruned_metrics
        X_matrix = X_matrix[:, ranked_knob_idxs]
        y_matrix = y_matrix[:, pruned_metric_idxs]

        # Combine duplicate rows (rows with same knob settings)
        X_matrix, y_matrix, rowlabels = DataUtil.combine_duplicate_rows(
            X_matrix, y_matrix, rowlabels)

        workload_data[unique_workload] = {
            'X_matrix': X_matrix,
            'y_matrix': y_matrix,
            'rowlabels': rowlabels,
        }

    # Stack all X & y matrices for preprocessing
    Xs = np.vstack([entry['X_matrix'] for entry in workload_data.values()])
    ys = np.vstack([entry['y_matrix'] for entry in workload_data.values()])

    # Scale the X & y values, then compute the deciles for each column in y
    X_scaler = StandardScaler(copy=False)
    X_scaler.fit(Xs)
    y_scaler = StandardScaler(copy=False)
    y_scaler.fit_transform(ys)
    y_binner = Bin(bin_start=1, axis=0)
    y_binner.fit(ys)
    del Xs
    del ys

    # Filter the target's X & y data by the ranked knobs & pruned metrics.
    X_target = target_data['X_matrix'][:, ranked_knob_idxs]
    y_target = target_data['y_matrix'][:, pruned_metric_idxs]

    # Now standardize the target's data and bin it by the deciles we just
    # calculated
    X_target = X_scaler.transform(X_target)
    y_target = y_scaler.transform(y_target)
    y_target = y_binner.transform(y_target)

    scores = {}
    for workload_id, workload_entry in workload_data.iteritems():
        predictions = np.empty_like(y_target)
        X_workload = workload_entry['X_matrix']
        y_workload = workload_entry['y_matrix']
        for j, y_col in enumerate(y_workload.T):
            # Using this workload's data, train a Gaussian process model
            # and then predict the performance of each metric for each of
            # the knob configurations attempted so far by the target.
            y_col = y_col.reshape(-1, 1)
            model = GPR()
            model.fit(X_workload, y_col, ridge=0.01)
            predictions[:, j] = model.predict(X_target).ypreds.ravel()
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
    for workload_id, similarity_score in scores.iteritems():
        if similarity_score < best_score:
            best_score = similarity_score
            best_workload_id = workload_id
    target_data['mapped_workload'] = (best_workload_id, best_score)
    target_data['scores'] = scores
    return target_data
