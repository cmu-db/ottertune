#
# OtterTune - periodic_tasks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import copy
import time
import numpy as np
from pytz import timezone

from celery import shared_task
from celery.utils.log import get_task_logger
from django.utils.timezone import now
from django.utils.datetime_safe import datetime
from sklearn.preprocessing import StandardScaler

from analysis.cluster import KMeansClusters, create_kselection_model
from analysis.factor_analysis import FactorAnalysis
from analysis.lasso import LassoPath
from analysis.preprocessing import (Bin, get_shuffle_indices,
                                    DummyEncoder,
                                    consolidate_columnlabels)
from website.models import PipelineData, PipelineRun, Result, Workload, ExecutionTime
from website.settings import (ENABLE_DUMMY_ENCODER, KNOB_IDENT_USE_PRUNED_METRICS,
                              MIN_WORKLOAD_RESULTS_COUNT, TIME_ZONE, VIEWS_FOR_PRUNING)
from website.types import PipelineTaskType, WorkloadStatusType
from website.utils import DataUtil, JSONUtil

# Log debug messages
LOG = get_task_logger(__name__)


def save_execution_time(start_ts, fn):
    end_ts = time.time()
    exec_time = end_ts - start_ts
    start_time = datetime.fromtimestamp(int(start_ts), timezone(TIME_ZONE))
    ExecutionTime.objects.create(module="celery.periodic_tasks", function=fn, tag="",
                                 start_time=start_time, execution_time=exec_time, result=None)

@shared_task(name="run_background_tasks")
def run_background_tasks():
    start_ts = time.time()
    LOG.info("Starting background tasks...")
    # Find modified and not modified workloads, we only have to calculate for the
    # modified workloads.
    modified_workloads = Workload.objects.filter(status=WorkloadStatusType.MODIFIED)
    num_modified = modified_workloads.count()
    non_modified_workloads = Workload.objects.filter(status=WorkloadStatusType.PROCESSED)
    non_modified_workloads = list(non_modified_workloads.values_list('pk', flat=True))
    last_pipeline_run = PipelineRun.objects.get_latest()
    LOG.debug("Workloads: # modified: %s, # processed: %s, # total: %s",
              num_modified, len(non_modified_workloads),
              Workload.objects.all().count())

    if num_modified == 0:
        # No previous workload data yet. Try again later.
        LOG.info("No modified workload data yet. Ending background tasks.")
        return

    # Create new entry in PipelineRun table to store the output of each of
    # the background tasks
    pipeline_run_obj = PipelineRun(start_time=now(), end_time=None)
    pipeline_run_obj.save()

    for i, workload in enumerate(modified_workloads):
        workload.status = WorkloadStatusType.PROCESSING
        workload.save()
        wkld_results = Result.objects.filter(workload=workload)
        num_wkld_results = wkld_results.count()
        workload_name = '{}@{}.{}'.format(workload.dbms.key, workload.project.name, workload.name)

        LOG.info("Starting workload %s (%s/%s, # results: %s)...", workload_name,
                 i + 1, num_modified, num_wkld_results)

        if num_wkld_results == 0:
            # delete the workload
            LOG.info("Deleting workload %s because it has no results.", workload_name)
            workload.delete()
            continue

        if num_wkld_results < MIN_WORKLOAD_RESULTS_COUNT:
            # Check that there are enough results in the workload
            LOG.info("Not enough results in workload %s (# results: %s, # required: %s).",
                     workload_name, num_wkld_results, MIN_WORKLOAD_RESULTS_COUNT)
            workload.status = WorkloadStatusType.PROCESSED
            workload.save()
            continue

        LOG.info("Aggregating data for workload %s...", workload_name)
        # Aggregate the knob & metric data for this workload
        knob_data, metric_data = aggregate_data(wkld_results)
        LOG.debug("Aggregated knob data: rowlabels=%s, columnlabels=%s, data=%s.",
                  len(knob_data['rowlabels']), len(knob_data['columnlabels']),
                  knob_data['data'].shape)
        LOG.debug("Aggregated metric data: rowlabels=%s, columnlabels=%s, data=%s.",
                  len(metric_data['rowlabels']), len(metric_data['columnlabels']),
                  metric_data['data'].shape)
        LOG.info("Done aggregating data for workload %s.", workload_name)

        num_valid_results = knob_data['data'].shape[0]  # pylint: disable=unsubscriptable-object
        if num_valid_results < MIN_WORKLOAD_RESULTS_COUNT:
            # Check that there are enough valid results in the workload
            LOG.info("Not enough valid results in workload %s (# valid results: "
                     "%s, # required: %s).", workload_name, num_valid_results,
                     MIN_WORKLOAD_RESULTS_COUNT)
            workload.status = WorkloadStatusType.PROCESSED
            workload.save()
            continue

        # Knob_data and metric_data are 2D numpy arrays. Convert them into a
        # JSON-friendly (nested) lists and then save them as new PipelineData
        # objects.
        knob_data_copy = copy.deepcopy(knob_data)
        knob_data_copy['data'] = knob_data_copy['data'].tolist()
        knob_data_copy = JSONUtil.dumps(knob_data_copy)
        knob_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                  task_type=PipelineTaskType.KNOB_DATA,
                                  workload=workload,
                                  data=knob_data_copy,
                                  creation_time=now())
        knob_entry.save()

        metric_data_copy = copy.deepcopy(metric_data)
        metric_data_copy['data'] = metric_data_copy['data'].tolist()
        metric_data_copy = JSONUtil.dumps(metric_data_copy)
        metric_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                    task_type=PipelineTaskType.METRIC_DATA,
                                    workload=workload,
                                    data=metric_data_copy,
                                    creation_time=now())
        metric_entry.save()

        # Execute the Workload Characterization task to compute the list of
        # pruned metrics for this workload and save them in a new PipelineData
        # object.
        LOG.info("Pruning metrics for workload %s...", workload_name)
        pruned_metrics = run_workload_characterization(metric_data=metric_data, dbms=workload.dbms)
        LOG.info("Done pruning metrics for workload %s (# pruned metrics: %s).\n\n"
                 "Pruned metrics: %s\n", workload_name, len(pruned_metrics),
                 pruned_metrics)
        pruned_metrics_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                            task_type=PipelineTaskType.PRUNED_METRICS,
                                            workload=workload,
                                            data=JSONUtil.dumps(pruned_metrics),
                                            creation_time=now())
        pruned_metrics_entry.save()

        # Workload target objective data
        ranked_knob_metrics = sorted(wkld_results.distinct('session').values_list(
            'session__target_objective', flat=True).distinct())
        LOG.debug("Target objectives for workload %s: %s", workload_name,
                  ', '.join(ranked_knob_metrics))

        if KNOB_IDENT_USE_PRUNED_METRICS:
            ranked_knob_metrics = sorted(set(ranked_knob_metrics) + set(pruned_metrics))

        # Use the set of metrics to filter the metric_data
        metric_idxs = [i for i, metric_name in enumerate(metric_data['columnlabels'])
                       if metric_name in ranked_knob_metrics]
        ranked_metric_data = {
            'data': metric_data['data'][:, metric_idxs],
            'rowlabels': copy.deepcopy(metric_data['rowlabels']),
            'columnlabels': [metric_data['columnlabels'][i] for i in metric_idxs]
        }

        # Execute the Knob Identification task to compute an ordered list of knobs
        # ranked by their impact on the DBMS's performance. Save them in a new
        # PipelineData object.
        LOG.info("Ranking knobs for workload %s (use pruned metric data: %s)...",
                 workload_name, KNOB_IDENT_USE_PRUNED_METRICS)
        sessions = []
        for result in wkld_results:
            if result.session not in sessions:
                sessions.append(result.session)
        rank_knob_data = copy.deepcopy(knob_data)
        rank_knob_data['data'], rank_knob_data['columnlabels'] =\
            DataUtil.clean_knob_data(knob_data['data'], knob_data['columnlabels'], sessions)
        ranked_knobs = run_knob_identification(knob_data=rank_knob_data,
                                               metric_data=ranked_metric_data,
                                               dbms=workload.dbms)
        LOG.info("Done ranking knobs for workload %s (# ranked knobs: %s).\n\n"
                 "Ranked knobs: %s\n", workload_name, len(ranked_knobs), ranked_knobs)
        ranked_knobs_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                          task_type=PipelineTaskType.RANKED_KNOBS,
                                          workload=workload,
                                          data=JSONUtil.dumps(ranked_knobs),
                                          creation_time=now())
        ranked_knobs_entry.save()

        workload.status = WorkloadStatusType.PROCESSED
        workload.save()
        LOG.info("Done processing workload %s (%s/%s).", workload_name, i + 1,
                 num_modified)

    LOG.info("Finished processing %s modified workloads.", num_modified)

    non_modified_workloads = Workload.objects.filter(pk__in=non_modified_workloads)
    # Update the latest pipeline data for the non modified workloads to have this pipeline run
    PipelineData.objects.filter(workload__in=non_modified_workloads,
                                pipeline_run=last_pipeline_run)\
        .update(pipeline_run=pipeline_run_obj)

    # Set the end_timestamp to the current time to indicate that we are done running
    # the background tasks
    pipeline_run_obj.end_time = now()
    pipeline_run_obj.save()
    save_execution_time(start_ts, "run_background_tasks")
    LOG.info("Finished background tasks (%.0f seconds).", time.time() - start_ts)


def aggregate_data(wkld_results):
    # Aggregates both the knob & metric data for the given workload.
    #
    # Parameters:
    #   wkld_results: result data belonging to this specific workload
    #
    # Returns: two dictionaries containing the knob & metric data as
    # a tuple

    # Now call the aggregate_data helper function to combine all knob &
    # metric data into matrices and also create row/column labels
    # (see the DataUtil class in website/utils.py)
    #
    # The aggregate_data helper function returns a dictionary of the form:
    #   - 'X_matrix': the knob data as a 2D numpy matrix (results x knobs)
    #   - 'y_matrix': the metric data as a 2D numpy matrix (results x metrics)
    #   - 'rowlabels': list of result ids that correspond to the rows in
    #         both X_matrix & y_matrix
    #   - 'X_columnlabels': a list of the knob names corresponding to the
    #         columns in the knob_data matrix
    #   - 'y_columnlabels': a list of the metric names corresponding to the
    #         columns in the metric_data matrix
    start_ts = time.time()
    aggregated_data = DataUtil.aggregate_data(wkld_results, ignore=['range_test', 'default', '*'])

    # Separate knob & workload data into two "standard" dictionaries of the
    # same form
    knob_data = {
        'data': aggregated_data['X_matrix'],
        'rowlabels': aggregated_data['rowlabels'],
        'columnlabels': aggregated_data['X_columnlabels']
    }

    metric_data = {
        'data': aggregated_data['y_matrix'],
        'rowlabels': copy.deepcopy(aggregated_data['rowlabels']),
        'columnlabels': aggregated_data['y_columnlabels']
    }

    # Return the knob & metric data
    save_execution_time(start_ts, "aggregate_data")
    return knob_data, metric_data


def run_workload_characterization(metric_data, dbms=None):
    # Performs workload characterization on the metric_data and returns
    # a set of pruned metrics.
    #
    # Parameters:
    #   metric_data is a dictionary of the form:
    #     - 'data': 2D numpy matrix of metric data (results x metrics)
    #     - 'rowlabels': a list of identifiers for the rows in the matrix
    #     - 'columnlabels': a list of the metric names corresponding to
    #                       the columns in the data matrix
    start_ts = time.time()

    matrix = metric_data['data']
    columnlabels = metric_data['columnlabels']
    LOG.debug("Workload characterization ~ initial data size: %s", matrix.shape)

    views = None if dbms is None else VIEWS_FOR_PRUNING.get(dbms.type, None)
    matrix, columnlabels = DataUtil.clean_metric_data(matrix, columnlabels, views)
    LOG.debug("Workload characterization ~ cleaned data size: %s", matrix.shape)

    # Bin each column (metric) in the matrix by its decile
    binner = Bin(bin_start=1, axis=0)
    binned_matrix = binner.fit_transform(matrix)

    # Remove any constant columns
    nonconst_matrix = []
    nonconst_columnlabels = []
    for col, cl in zip(binned_matrix.T, columnlabels):
        if np.any(col != col[0]):
            nonconst_matrix.append(col.reshape(-1, 1))
            nonconst_columnlabels.append(cl)
    assert len(nonconst_matrix) > 0, "Need more data to train the model"
    nonconst_matrix = np.hstack(nonconst_matrix)
    LOG.debug("Workload characterization ~ nonconst data size: %s", nonconst_matrix.shape)

    # Remove any duplicate columns
    unique_matrix, unique_idxs = np.unique(nonconst_matrix, axis=1, return_index=True)
    unique_columnlabels = [nonconst_columnlabels[idx] for idx in unique_idxs]

    LOG.debug("Workload characterization ~ final data size: %s", unique_matrix.shape)
    n_rows, n_cols = unique_matrix.shape

    # Shuffle the matrix rows
    shuffle_indices = get_shuffle_indices(n_rows)
    shuffled_matrix = unique_matrix[shuffle_indices, :]

    # Fit factor analysis model
    fa_model = FactorAnalysis()
    # For now we use 5 latent variables
    fa_model.fit(shuffled_matrix, unique_columnlabels, n_components=5)

    # Components: metrics * factors
    components = fa_model.components_.T.copy()
    LOG.info("Workload characterization first part costs %.0f seconds.", time.time() - start_ts)

    # Run Kmeans for # clusters k in range(1, num_nonduplicate_metrics - 1)
    # K should be much smaller than n_cols in detK, For now max_cluster <= 20
    kmeans_models = KMeansClusters()
    kmeans_models.fit(components, min_cluster=1,
                      max_cluster=min(n_cols - 1, 20),
                      sample_labels=unique_columnlabels,
                      estimator_params={'n_init': 50})

    # Compute optimal # clusters, k, using gap statistics
    gapk = create_kselection_model("gap-statistic")
    gapk.fit(components, kmeans_models.cluster_map_)

    LOG.debug("Found optimal number of clusters: %d", gapk.optimal_num_clusters_)

    # Get pruned metrics, cloest samples of each cluster center
    pruned_metrics = kmeans_models.cluster_map_[gapk.optimal_num_clusters_].get_closest_samples()

    # Return pruned metrics
    save_execution_time(start_ts, "run_workload_characterization")
    LOG.info("Workload characterization finished in %.0f seconds.", time.time() - start_ts)
    return pruned_metrics


def run_knob_identification(knob_data, metric_data, dbms):
    # Performs knob identification on the knob & metric data and returns
    # a set of ranked knobs.
    #
    # Parameters:
    #   knob_data & metric_data are dictionaries of the form:
    #     - 'data': 2D numpy matrix of knob/metric data
    #     - 'rowlabels': a list of identifiers for the rows in the matrix
    #     - 'columnlabels': a list of the knob/metric names corresponding
    #           to the columns in the data matrix
    #   dbms is the foreign key pointing to target dbms in DBMSCatalog
    #
    # When running the lasso algorithm, the knob_data matrix is set of
    # independent variables (X) and the metric_data is the set of
    # dependent variables (y).
    start_ts = time.time()

    knob_matrix = knob_data['data']
    knob_columnlabels = knob_data['columnlabels']

    metric_matrix = metric_data['data']
    metric_columnlabels = metric_data['columnlabels']

    # remove constant columns from knob_matrix and metric_matrix
    nonconst_knob_matrix = []
    nonconst_knob_columnlabels = []

    for col, cl in zip(knob_matrix.T, knob_columnlabels):
        if np.any(col != col[0]):
            nonconst_knob_matrix.append(col.reshape(-1, 1))
            nonconst_knob_columnlabels.append(cl)
    assert len(nonconst_knob_matrix) > 0, "Need more data to train the model"
    nonconst_knob_matrix = np.hstack(nonconst_knob_matrix)

    nonconst_metric_matrix = []
    nonconst_metric_columnlabels = []

    for col, cl in zip(metric_matrix.T, metric_columnlabels):
        if np.any(col != col[0]):
            nonconst_metric_matrix.append(col.reshape(-1, 1))
            nonconst_metric_columnlabels.append(cl)
    nonconst_metric_matrix = np.hstack(nonconst_metric_matrix)

    if ENABLE_DUMMY_ENCODER:
        # determine which knobs need encoding (enums with >2 possible values)

        categorical_info = DataUtil.dummy_encoder_helper(nonconst_knob_columnlabels,
                                                         dbms)
        # encode categorical variable first (at least, before standardize)
        dummy_encoder = DummyEncoder(categorical_info['n_values'],
                                     categorical_info['categorical_features'],
                                     categorical_info['cat_columnlabels'],
                                     categorical_info['noncat_columnlabels'])
        encoded_knob_matrix = dummy_encoder.fit_transform(
            nonconst_knob_matrix)
        encoded_knob_columnlabels = dummy_encoder.new_labels
    else:
        encoded_knob_columnlabels = nonconst_knob_columnlabels
        encoded_knob_matrix = nonconst_knob_matrix

    # standardize values in each column to N(0, 1)
    standardizer = StandardScaler()
    standardized_knob_matrix = standardizer.fit_transform(encoded_knob_matrix)
    standardized_metric_matrix = standardizer.fit_transform(nonconst_metric_matrix)

    # shuffle rows (note: same shuffle applied to both knob and metric matrices)
    shuffle_indices = get_shuffle_indices(standardized_knob_matrix.shape[0], seed=17)
    shuffled_knob_matrix = standardized_knob_matrix[shuffle_indices, :]
    shuffled_metric_matrix = standardized_metric_matrix[shuffle_indices, :]

    # run lasso algorithm
    lasso_model = LassoPath()
    lasso_model.fit(shuffled_knob_matrix, shuffled_metric_matrix, encoded_knob_columnlabels)

    # consolidate categorical feature columns, and reset to original names
    encoded_knobs = lasso_model.get_ranked_features()
    consolidated_knobs = consolidate_columnlabels(encoded_knobs)

    save_execution_time(start_ts, "run_knob_identification")
    LOG.info("Knob identification finished in %.0f seconds.", time.time() - start_ts)
    return consolidated_knobs
