import itertools
import numpy as np
import os.path
from collections import OrderedDict

from celery.task import task, Task
from django.utils.timezone import now
from djcelery.models import TaskMeta
from sklearn.preprocessing import StandardScaler

from analysis.gp_tf import GPR, GPR_GD
from analysis.preprocessing import bin_by_decile, Bin
from website.models import (DBConf, DBMSCatalog, Hardware, KnobCatalog, PipelineResult,
                            Result, Workload)
from website.settings import PIPELINE_DIR
from website.types import KnobUnitType, PipelineTaskType, VarType
from website.utils import (ConversionUtil, DataUtil, DBMSUtil, JSONUtil,
                           MediaUtil, PostgresUtilImpl)





class UpdateTask(Task):

    def __init__(self):
        self.rate_limit = '50/m'
        self.max_retries = 3
        self.default_retry_delay = 60


class AggregateTargetResults(UpdateTask):

    def on_success(self, retval, task_id, args, kwargs):
        super(UpdateTask, self).on_success(retval, task_id, args, kwargs)

        # Completely delete this result because it's huge and not
        # interesting
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = None
        task_meta.save()


class MapWorkload(UpdateTask):

    def on_success(self, retval, task_id, args, kwargs):
        super(UpdateTask, self).on_success(retval, task_id, args, kwargs)

        # Replace result with formatted result
        new_res = {
            'scores': sorted(args[0]['scores'].iteritems()),
            'mapped_workload_id': args[0]['mapped_workload'],
        }
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = new_res # Only store scores
        task_meta.save()


class ConfigurationRecommendation(UpdateTask):

    def on_success(self, retval, task_id, args, kwargs):
        super(UpdateTask, self).on_success(retval, task_id, args, kwargs)

        result_id = args[0]['newest_result_id']
        result = Result.objects.get(pk=result_id)

        # Replace result with formatted result
        formatted_params = DBMSUtil.format_dbms_params(result.dbms.pk, retval)
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = formatted_params
        task_meta.save()

        # Create next configuration to try
        nondefault_params = JSONUtil.loads(
            result.application.nondefault_settings)
        config = DBMSUtil.create_configuration(
            result.dbms.pk, formatted_params, nondefault_params)
        path_prefix = MediaUtil.get_result_data_path(result.pk)
        path = '{}.next_conf'.format(path_prefix)
        with open(path, 'w') as f:
            f.write(config)

@task(base=AggregateTargetResults, name='aggregate_target_results')
def aggregate_target_results(result_id):
    newest_result = Result.objects.get(pk=result_id)
    target_results = Result.objects.filter(
        application=newest_result.application, dbms=newest_result.dbms)
    if len(target_results) == 0:
        raise Exception('Cannot find any results for app_id={}, dbms_id={}'
                        .format(newest_result.application, newest_result.dbms))
    agg_data = DataUtil.aggregate_data(target_results)

    agg_data['newest_result_id'] = result_id
    return agg_data


@task(base=ConfigurationRecommendation, name='configuration_recommendation')
def configuration_recommendation(target_data):
    if target_data['scores'] is None:
        raise NotImplementedError('Implement me!')
    best_wkld_id = target_data['mapped_workload'][0]

    # Load specific workload data
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    target_obj = newest_result.application.target_objective
    dbms_id = newest_result.dbms.pk
    hw_id = newest_result.application.hardware.pk
    agg_data = PipelineResult.get_latest(
        dbms_id, hw_id, PipelineTaskType.AGGREGATED_DATA)
    if agg_data is None:
        return None
    data_map = JSONUtil.loads(agg_data.value)
    if best_wkld_id not in data_map['data']:
        raise Exception(('Cannot find mapped workload'
                         '(id={}) in aggregated data').format(best_wkld_id))
    workload_data = np.load(data_map['data'][best_wkld_id])

    # Mapped workload data
    X_wkld_matrix = workload_data['X_matrix']
    y_wkld_matrix = workload_data['y_matrix']
    wkld_rowlabels = workload_data['rowlabels']
    X_columnlabels = workload_data['X_columnlabels']
    y_columnlabels = workload_data['y_columnlabels']

    # Target workload data
    X_target_matrix = target_data['X_matrix']
    y_target_matrix = target_data['y_matrix']
    target_rowlabels = target_data['rowlabels']

    if not np.array_equal(X_columnlabels, target_data['X_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical X columnlabels (sorted knob names)'))
    if not np.array_equal(y_columnlabels, target_data['y_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical y columnlabels (sorted metric names)'))

    # Filter knobs
    ranked_knobs = JSONUtil.loads(PipelineResult.get_latest(
        dbms_id, hw_id, PipelineTaskType.RANKED_KNOBS).value)[:10]  # FIXME
    X_idxs = [i for i in range(X_columnlabels.shape[0]) if X_columnlabels[
        i] in ranked_knobs]
    X_wkld_matrix = X_wkld_matrix[:, X_idxs]
    X_target_matrix = X_target_matrix[:, X_idxs]
    X_columnlabels = X_columnlabels[X_idxs]

    # Filter metrics by current target objective metric
    y_idx = [i for i in range(y_columnlabels.shape[0])
             if y_columnlabels[i] == target_obj]
    if len(y_idx) == 0:
        raise Exception(('Could not find target objective in metrics '
                         '(target_obj={})').format(target_obj))
    elif len(y_idx) > 1:
        raise Exception(('Found {} instances of target objective in '
                         'metrics (target_obj={})').format(len(y_idx),
                                                           target_obj))
    y_wkld_matrix = y_wkld_matrix[:, y_idx]
    y_target_matrix = y_target_matrix[:, y_idx]
    y_columnlabels = y_columnlabels[y_idx]

    # Combine duplicate rows in the target/workload data (separately)
    X_wkld_matrix, y_wkld_matrix, wkld_rowlabels = DataUtil.combine_duplicate_rows(
        X_wkld_matrix, y_wkld_matrix, wkld_rowlabels)
    X_target_matrix, y_target_matrix, target_rowlabels = DataUtil.combine_duplicate_rows(
        X_target_matrix, y_target_matrix, target_rowlabels)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    dups_filter = np.ones(X_wkld_matrix.shape[0], dtype=bool)
    target_row_tups = [tuple(row) for row in X_target_matrix]
    for i, row in enumerate(X_wkld_matrix):
        if tuple(row) in target_row_tups:
            dups_filter[i] = False
    X_wkld_matrix = X_wkld_matrix[dups_filter, :]
    y_wkld_matrix = y_wkld_matrix[dups_filter, :]
    wkld_rowlabels = wkld_rowlabels[dups_filter]

    # Combine Xs and scale
    X_matrix = np.vstack([X_target_matrix, X_wkld_matrix])
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target_matrix.shape[0] < 5:  # FIXME
        y_target_scaler = None
        y_wkld_scaler = StandardScaler()
        y_matrix = np.vstack([y_target_matrix, y_wkld_matrix])
        y_scaled = y_wkld_scaler.fit_transform(y_matrix)
    else:
        try:
            y_target_scaler = StandardScaler()
            y_wkld_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target_matrix)
            y_wkld_scaled = y_wkld_scaler.fit_transform(y_wkld_matrix)
            y_scaled = np.vstack([y_target_scaled, y_wkld_scaled])
        except ValueError:
            y_target_scaler = None
            y_wkld_scaler = StandardScaler()
            y_matrix = np.vstack([y_target_matrix, y_wkld_matrix])
            y_scaled = y_wkld_scaler.fit_transform(y_matrix)

    ridge = np.empty(X_scaled.shape[0])
    ridge[:X_target_matrix.shape[0]] = 0.01
    ridge[X_target_matrix.shape[0]:] = 0.1

    # FIXME
    num_samples = 5
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    for i in range(X_scaled.shape[1]):
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        X_samples[:, i] = np.random.rand(
            num_samples) * (col_max - col_min) + col_min

    model = GPR_GD()
    model.fit(X_scaled, y_scaled, ridge)
    res = model.predict(X_samples)
    best_idx = np.argmin(res.minL.ravel())
    best_conf = res.minL_conf[best_idx, :]
    best_conf = X_scaler.inverse_transform(best_conf)

    conf_map = {k: best_conf[i] for i,k in enumerate(X_columnlabels)}
    return conf_map


@task(base=MapWorkload, name='map_workload')
def map_workload(target_data):
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    dbms = newest_result.dbms.pk
    hardware = newest_result.application.hardware.pk
    workload_data = PipelineResult.get_latest(
        dbms, hardware, PipelineTaskType.WORKLOAD_MAPPING_DATA)
    if workload_data is None:
        target_data['scores'] = None
        return target_data

    data_values = JSONUtil.loads(workload_data.value)
    X_scaler = np.load(data_values['X_scaler'])
    y_scaler = np.load(data_values['y_scaler'])
    y_deciles = np.load(data_values['y_deciles'])['deciles']
    X_columnlabels = data_values['X_columnlabels']
    y_columnlabels = data_values['y_columnlabels']

    X_idxs = [i for i in range(target_data['X_matrix'].shape[1]) if target_data[
        'X_columnlabels'][i] in X_columnlabels]
    y_idxs = [i for i in range(target_data['y_matrix'].shape[1]) if target_data[
        'y_columnlabels'][i] in y_columnlabels]
    X_target = target_data['X_matrix'][:, X_idxs]
    y_target = target_data['y_matrix'][:, y_idxs]
    X_target = (X_target - X_scaler['mean']) / X_scaler['scale']
    y_target = (y_target - y_scaler['mean']) / y_scaler['scale']
    y_binned = np.empty_like(y_target)
    for i in range(y_target.shape[1]):
        y_binned[:, i] = bin_by_decile(y_target[:, i], y_deciles[i])

    scores = {}
    for wkld_id, wkld_entry_path in data_values['data'].iteritems():
        wkld_entry = np.load(wkld_entry_path)
        preds = np.empty_like(y_target)
        X_wkld = wkld_entry['X_matrix']
        for j in range(y_target.shape[1]):
            y_col = wkld_entry['y_matrix'][:, j].reshape(X_wkld.shape[0], 1)
            model = GPR()
            model.fit(X_wkld, y_col, ridge=0.01)
            preds[:, j] = bin_by_decile(model.predict(
                X_target).ypreds.ravel(), y_deciles[j])
        dists = np.sqrt(
            np.sum(np.square(np.subtract(preds, y_target)), axis=1))
        scores[wkld_id] = np.mean(dists)

    # Find the best (minimum) score
    best_score = np.inf
    best_wkld_id = None
    for wkld_id, similarity_score in scores.iteritems():
        if similarity_score < best_score:
            best_score = similarity_score
            best_wkld_id = wkld_id
    target_data['mapped_workload'] = (best_wkld_id, best_score)
    target_data['scores'] = scores
    return target_data


@task(name='aggregate_results')
def aggregate_results():
    unique_workloads = Workload.objects.all()
    all_data = {}
    for workload in unique_workloads:
        results = Result.objects.filter(workload=workload)
        if len(results) < 2:
            continue
        entry = DataUtil.aggregate_data(results)
        key = (workload.dbms.pk, workload.hardware.pk)
        if key not in all_data:
            all_data[key] = {}
        all_data[key][workload.pk] = entry

    ts = now()
    tsf = ts.strftime("%Y%m%d-%H%M%S")
    for (dbkey, hwkey), workload_data in all_data.iteritems():
        task_name = PipelineTaskType.TYPE_NAMES[
            PipelineTaskType.AGGREGATED_DATA].replace(' ', '').upper()
        savepaths = {}
        for workloadkey, entry in workload_data.iteritems():
            fname = '{}_{}_{}_{}_{}.npz'.format(
                task_name, dbkey, hwkey, workloadkey, tsf)
            savepath = os.path.join(PIPELINE_DIR, fname)
            savepaths[workloadkey] = savepath
            np.savez_compressed(savepath, **entry)

        value = {
            'data': savepaths
        }

        new_res = PipelineResult()
        new_res.dbms = DBMSCatalog.objects.get(pk=dbkey)
        new_res.hardware = Hardware.objects.get(pk=hwkey)
        new_res.creation_timestamp = ts
        new_res.task_type = PipelineTaskType.AGGREGATED_DATA
        new_res.value = JSONUtil.dumps(value)
        new_res.save()


@task(name='create_workload_mapping_data')
def create_workload_mapping_data():
    agg_datas = PipelineResult.objects.filter(
        task_type=PipelineTaskType.AGGREGATED_DATA)
    dbmss = set([ad.dbms.pk for ad in agg_datas])
    hardwares = set([ad.hardware.pk for ad in agg_datas])

    for dbms_id, hw_id in itertools.product(dbmss, hardwares):
        data = PipelineResult.get_latest(
            dbms_id, hw_id, PipelineTaskType.AGGREGATED_DATA)
        file_info = JSONUtil.loads(data.value)
        cluster_data = OrderedDict()
        for cluster, path in file_info['data'].iteritems():
            compressed_data = np.load(path)
            X_matrix = compressed_data['X_matrix']
            y_matrix = compressed_data['y_matrix']
            X_columnlabels = compressed_data['X_columnlabels']
            y_columnlabels = compressed_data['y_columnlabels']
            rowlabels = compressed_data['rowlabels']

            # Filter metrics and knobs
            ranked_knobs = JSONUtil.loads(PipelineResult.get_latest(
                dbms_id, hw_id, PipelineTaskType.RANKED_KNOBS).value)[:10]  # FIXME
            pruned_metrics = JSONUtil.loads(PipelineResult.get_latest(
                dbms_id, hw_id, PipelineTaskType.PRUNED_METRICS).value)
            knob_idxs = [i for i in range(X_matrix.shape[1]) if X_columnlabels[
                i] in ranked_knobs]
            metric_idxs = [i for i in range(y_matrix.shape[1]) if y_columnlabels[
                i] in pruned_metrics]
            X_matrix = X_matrix[:, knob_idxs]
            X_columnlabels = X_columnlabels[knob_idxs]
            y_matrix = y_matrix[:, metric_idxs]
            y_columnlabels = y_columnlabels[metric_idxs]

            # Combine duplicate rows
            X_matrix, y_matrix, rowlabels = DataUtil.combine_duplicate_rows(
                X_matrix, y_matrix, rowlabels)
            cluster_data[cluster] = {
                'X_matrix': X_matrix,
                'y_matrix': y_matrix,
                'X_columnlabels': X_columnlabels,
                'y_columnlabels': y_columnlabels,
                'rowlabels': rowlabels,
            }

        Xs = np.vstack([entry['X_matrix'] for entry in cluster_data.values()])
        ys = np.vstack([entry['y_matrix'] for entry in cluster_data.values()])

        X_scaler = StandardScaler(copy=False)
        X_scaler.fit(Xs)
        y_scaler = StandardScaler(copy=False)
        y_scaler.fit_transform(ys)
        y_binner = Bin(axis=0)
        y_binner.fit(ys)
        del Xs
        del ys

        task_name = PipelineTaskType.TYPE_NAMES[
            PipelineTaskType.WORKLOAD_MAPPING_DATA].replace(' ', '').upper()
        timestamp = data.creation_timestamp
        tsf = timestamp.strftime("%Y%m%d-%H%M%S")
        savepaths = {}
        for cluster, entry in cluster_data.iteritems():
            X_scaler.transform(entry['X_matrix'])
            y_scaler.transform(entry['y_matrix'])
            fname = '{}_{}_{}_{}_{}.npz'.format(
                task_name, dbms_id, hw_id, cluster, tsf)
            savepath = os.path.join(PIPELINE_DIR, fname)
            savepaths[cluster] = savepath
            np.savez_compressed(savepath, **entry)

        X_scaler_path = os.path.join(
            PIPELINE_DIR, '{}_XSCALER_{}_{}_{}.npz'.format(task_name, dbms_id, hw_id, tsf))
        np.savez_compressed(
            X_scaler_path, mean=X_scaler.mean_, scale=X_scaler.scale_)
        y_scaler_path = os.path.join(
            PIPELINE_DIR, '{}_YSCALER_{}_{}_{}.npz'.format(task_name, dbms_id, hw_id, tsf))
        np.savez_compressed(
            y_scaler_path, mean=y_scaler.mean_, scale=y_scaler.scale_)
        y_deciles_path = os.path.join(
            PIPELINE_DIR, '{}_YDECILES_{}_{}_{}.npz'.format(task_name, dbms_id, hw_id, tsf))
        np.savez_compressed(y_deciles_path, deciles=y_binner.deciles_)

        value = {
            'data': savepaths,
            'X_scaler': X_scaler_path,
            'y_scaler': y_scaler_path,
            'y_deciles': y_deciles_path,
            'X_columnlabels': cluster_data.values()[0]['X_columnlabels'].tolist(),
            'y_columnlabels': cluster_data.values()[0]['y_columnlabels'].tolist(),
        }

        new_res = PipelineResult()
        new_res.dbms = DBMSCatalog.objects.get(pk=dbms_id)
        new_res.hardware = Hardware.objects.get(pk=hw_id)
        new_res.creation_timestamp = timestamp
        new_res.task_type = PipelineTaskType.WORKLOAD_MAPPING_DATA
        new_res.value = JSONUtil.dumps(value, pprint=True)
        new_res.save()





