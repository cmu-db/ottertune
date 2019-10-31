#
# OtterTune - models.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from collections import OrderedDict

from django.contrib.auth.models import User
from django.db import models, DEFAULT_DB_ALIAS
from django.utils.timezone import now

from .db import target_objectives
from .types import (DBMSType, LabelStyleType, MetricType, KnobUnitType,
                    PipelineTaskType, VarType, KnobResourceType,
                    WorkloadStatusType, AlgorithmType, StorageType)


class BaseModel(models.Model):

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        return getattr(self, 'name', str(self.pk))

    @classmethod
    def get_labels(cls, style=LabelStyleType.DEFAULT_STYLE):
        from .utils import LabelUtil

        labels = {}
        fields = cls._meta.get_fields()
        for field in fields:
            try:
                verbose_name = field.verbose_name
                if field.name == 'id':
                    verbose_name = cls._model_name() + ' id'
                labels[field.name] = verbose_name
            except AttributeError:
                pass
        return LabelUtil.style_labels(labels, style)

    @classmethod
    def _model_name(cls):
        return cls.__name__

    class Meta:  # pylint: disable=no-init
        abstract = True


class DBMSCatalog(BaseModel):
    type = models.IntegerField(choices=DBMSType.choices())
    version = models.CharField(max_length=16)

    @property
    def name(self):
        return DBMSType.name(self.type)

    @property
    def key(self):
        return '{}_{}'.format(self.name, self.version)

    @property
    def full_name(self):
        return '{} v{}'.format(self.name, self.version)

    def __unicode__(self):
        return self.full_name


class KnobCatalog(BaseModel):
    dbms = models.ForeignKey(DBMSCatalog)
    name = models.CharField(max_length=128)
    vartype = models.IntegerField(choices=VarType.choices(), verbose_name="variable type")
    unit = models.IntegerField(choices=KnobUnitType.choices())
    category = models.TextField(null=True)
    summary = models.TextField(null=True, verbose_name='description')
    description = models.TextField(null=True)
    scope = models.CharField(max_length=16)
    minval = models.CharField(max_length=32, null=True, verbose_name="minimum value")
    maxval = models.CharField(max_length=32, null=True, verbose_name="maximum value")
    default = models.TextField(verbose_name="default value")
    enumvals = models.TextField(null=True, verbose_name="valid values")
    context = models.CharField(max_length=32)
    tunable = models.BooleanField(verbose_name="tunable")
    resource = models.IntegerField(choices=KnobResourceType.choices(), default=4)


class MetricCatalog(BaseModel):
    dbms = models.ForeignKey(DBMSCatalog)
    name = models.CharField(max_length=128)
    vartype = models.IntegerField(choices=VarType.choices())
    summary = models.TextField(null=True, verbose_name='description')
    scope = models.CharField(max_length=16)
    metric_type = models.IntegerField(choices=MetricType.choices())


class Project(BaseModel):
    user = models.ForeignKey(User)
    name = models.CharField(max_length=64, verbose_name="project name")
    description = models.TextField(null=True, blank=True)
    creation_time = models.DateTimeField()
    last_update = models.DateTimeField()

    def delete(self, using=DEFAULT_DB_ALIAS, keep_parents=False):
        sessions = Session.objects.filter(project=self)
        for x in sessions:
            x.delete()
        super(Project, self).delete(using, keep_parents)

    class Meta:  # pylint: disable=no-init
        unique_together = ('user', 'name')


class Hardware(BaseModel):

    @property
    def name(self):
        return '{} CPUs, {}GB RAM, {}GB {}'.format(
            self.cpu, self.memory, self.storage, StorageType.name(self.storage_type))

    cpu = models.IntegerField(default=4, verbose_name='Number of CPUs')
    memory = models.IntegerField(default=16, verbose_name='Memory (GB)')
    storage = models.IntegerField(default=32, verbose_name='Storage (GB)')
    storage_type = models.IntegerField(choices=StorageType.choices(),
                                       default=StorageType.SSD, verbose_name='Storage Type')
    additional_specs = models.TextField(null=True, default=None)

    class Meta:  # pylint: disable=no-init
        unique_together = ('cpu', 'memory', 'storage', 'storage_type')


class Session(BaseModel):

    TUNING_OPTIONS = OrderedDict([
        ("tuning_session", "Tuning Session"),
        ("no_tuning_session", "No Tuning"),
        ("randomly_generate", "Randomly Generate"),
        ("lhs", "Run LHS")
    ])

    user = models.ForeignKey(User)
    name = models.CharField(max_length=64, verbose_name="session name")
    description = models.TextField(null=True, blank=True)
    dbms = models.ForeignKey(DBMSCatalog)
    hardware = models.ForeignKey(Hardware)
    algorithm = models.IntegerField(choices=AlgorithmType.choices(),
                                    default=AlgorithmType.GPR)
    lhs_samples = models.TextField(default="[]")
    ddpg_actor_model = models.BinaryField(null=True, blank=True)
    ddpg_critic_model = models.BinaryField(null=True, blank=True)
    ddpg_reply_memory = models.BinaryField(null=True, blank=True)
    dnn_model = models.BinaryField(null=True, blank=True)

    project = models.ForeignKey(Project)
    creation_time = models.DateTimeField()
    last_update = models.DateTimeField()

    upload_code = models.CharField(max_length=30, unique=True)
    tuning_session = models.CharField(choices=TUNING_OPTIONS.items(),
                                      max_length=64, default='tuning_session',
                                      verbose_name='session type')

    target_objective = models.CharField(
        max_length=64, default=target_objectives.default())

    def clean(self):
        if self.target_objective is None:
            self.target_objective = target_objectives.default()

    def delete(self, using=DEFAULT_DB_ALIAS, keep_parents=False):
        SessionKnob.objects.get(session=self).delete()
        results = Result.objects.filter(session=self)
        for r in results:
            r.knob_data.delete()
            r.metric_data.delete()
            r.delete()
        super(Session, self).delete(using=DEFAULT_DB_ALIAS, keep_parents=False)

    class Meta:  # pylint: disable=no-init
        unique_together = ('user', 'project', 'name')


class SessionKnobManager(models.Manager):
    @staticmethod
    def get_knobs_for_session(session):
        # Returns a dict of the knob
        knobs = KnobCatalog.objects.filter(dbms=session.dbms)
        knob_dicts = list(knobs.values())
        for i, _ in enumerate(knob_dicts):
            if SessionKnob.objects.filter(session=session, knob=knobs[i]).exists():
                new_knob = SessionKnob.objects.filter(session=session, knob=knobs[i])[0]
                knob_dicts[i]["minval"] = new_knob.minval
                knob_dicts[i]["maxval"] = new_knob.maxval
                knob_dicts[i]["tunable"] = new_knob.tunable
        knob_dicts = [knob for knob in knob_dicts if knob["tunable"]]
        return knob_dicts

    @staticmethod
    def get_knob_min_max_tunability(session):
        # Returns a dict of the knob
        knobs = KnobCatalog.objects.filter(dbms=session.dbms)
        knob_dicts = list(knobs.values())
        session_knob_dicts = {}
        for i, _ in enumerate(knob_dicts):
            if SessionKnob.objects.filter(session=session, knob=knobs[i]).exists():
                new_knob = SessionKnob.objects.filter(session=session, knob=knobs[i])[0]
                min_max_tunability = {"minval": new_knob.minval,
                                      "maxval": new_knob.maxval,
                                      "tunable": new_knob.tunable}
                session_knob_dicts[new_knob.name] = min_max_tunability
        return session_knob_dicts

    @staticmethod
    def set_knob_min_max_tunability(session, knob_dicts):
        # Returns a dict of the knob
        session_knobs = SessionKnob.objects.filter(session=session)
        for session_knob in session_knobs:
            if knob_dicts.__contains__(session_knob.name):
                session_knob.minval = knob_dicts[session_knob.name]["minval"]
                session_knob.maxval = knob_dicts[session_knob.name]["maxval"]
                session_knob.tunable = knob_dicts[session_knob.name]["tunable"]
                session_knob.save()


class SessionKnob(BaseModel):

    @property
    def name(self):
        return self.knob.name

    objects = SessionKnobManager()
    session = models.ForeignKey(Session)
    knob = models.ForeignKey(KnobCatalog)
    minval = models.CharField(max_length=32, null=True, verbose_name="minimum value")
    maxval = models.CharField(max_length=32, null=True, verbose_name="maximum value")
    tunable = models.BooleanField(verbose_name="tunable")


class DataModel(BaseModel):
    session = models.ForeignKey(Session)
    name = models.CharField(max_length=50)
    creation_time = models.DateTimeField()
    data = models.TextField()
    dbms = models.ForeignKey(DBMSCatalog)

    class Meta:  # pylint: disable=no-init
        abstract = True


class DataManager(models.Manager):

    @staticmethod
    def create_name(data_obj, key):
        ts = data_obj.creation_time.strftime("%m-%d-%y")
        return key + '@' + ts + '#' + str(data_obj.pk)


class KnobDataManager(DataManager):

    def create_knob_data(self, session, knobs, data, dbms):
        try:
            return KnobData.objects.get(session=session,
                                        knobs=knobs)
        except KnobData.DoesNotExist:
            knob_data = self.create(session=session,
                                    knobs=knobs,
                                    data=data,
                                    dbms=dbms,
                                    creation_time=now())
            knob_data.name = self.create_name(knob_data, dbms.key)
            knob_data.save()
            return knob_data


class KnobData(DataModel):
    objects = KnobDataManager()

    knobs = models.TextField()


class MetricDataManager(DataManager):

    def create_metric_data(self, session, metrics, data, dbms):
        metric_data = self.create(session=session,
                                  metrics=metrics,
                                  data=data,
                                  dbms=dbms,
                                  creation_time=now())
        metric_data.name = self.create_name(metric_data, dbms.key)
        metric_data.save()
        return metric_data


class MetricData(DataModel):
    objects = MetricDataManager()

    metrics = models.TextField()


class WorkloadManager(models.Manager):

    def create_workload(self, dbms, hardware, name):
        # (dbms,hardware,name) should be unique for each workload
        try:
            return Workload.objects.get(dbms=dbms, hardware=hardware, name=name)
        except Workload.DoesNotExist:
            return self.create(dbms=dbms,
                               hardware=hardware,
                               name=name)


class Workload(BaseModel):

    # __DEFAULT_FMT = '{db}_{hw}_UNASSIGNED'.format

    objects = WorkloadManager()

    dbms = models.ForeignKey(DBMSCatalog)
    hardware = models.ForeignKey(Hardware)
    name = models.CharField(max_length=128, verbose_name='workload name')
    status = models.IntegerField(choices=WorkloadStatusType.choices(),
                                 default=WorkloadStatusType.MODIFIED,
                                 editable=False)

    def delete(self, using=DEFAULT_DB_ALIAS, keep_parents=False):
        # The results should not have corresponding workloads.
        # results = Result.objects.filter(workload=self)
        # if results.exists():
        #     raise Exception("Cannot delete {} workload since results exist. ".format(self.name))

        # Delete PipelineData with corresponding workloads
        pipelinedatas = PipelineData.objects.filter(workload=self)
        for x in pipelinedatas:
            x.delete()
        super(Workload, self).delete(using, keep_parents)

    class Meta:  # pylint: disable=no-init
        unique_together = ("dbms", "hardware", "name")

    # @property
    # def isdefault(self):
    #     return self.cluster_name == self.default
    #
    # @property
    # def default(self):
    #     return self.__DEFAULT_FMT(db=self.dbms.pk,
    #                               hw=self.hardware.pk)
    #
    # @staticmethod
    # def get_default(dbms_id, hw_id):
    #     return Workload.__DEFAULT_FMT(db=dbms_id,
    #                                   hw=hw_id)


class ResultManager(models.Manager):

    def create_result(self, session, dbms, workload,
                      knob_data, metric_data,
                      observation_start_time,
                      observation_end_time,
                      observation_time,
                      task_ids=None,
                      next_config=None):
        return self.create(
            session=session,
            dbms=dbms,
            workload=workload,
            knob_data=knob_data,
            metric_data=metric_data,
            observation_start_time=observation_start_time,
            observation_end_time=observation_end_time,
            observation_time=observation_time,
            task_ids=task_ids,
            next_configuration=next_config,
            creation_time=now())


class Result(BaseModel):
    objects = ResultManager()

    session = models.ForeignKey(Session, verbose_name='session name')
    dbms = models.ForeignKey(DBMSCatalog)
    workload = models.ForeignKey(Workload)
    knob_data = models.ForeignKey(KnobData)
    metric_data = models.ForeignKey(MetricData)

    creation_time = models.DateTimeField()
    observation_start_time = models.DateTimeField()
    observation_end_time = models.DateTimeField()
    observation_time = models.FloatField()
    task_ids = models.CharField(max_length=180, null=True)
    next_configuration = models.TextField(null=True)

    def __unicode__(self):
        return str(self.pk)


class PipelineRunManager(models.Manager):

    def get_latest(self):
        return self.all().exclude(end_time=None).first()


class PipelineRun(models.Model):
    objects = PipelineRunManager()

    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True)

    def __unicode__(self):
        return str(self.pk)

    def __str__(self):
        return self.__unicode__()

    class Meta:  # pylint: disable=no-init
        ordering = ["-id"]


class PipelineData(models.Model):
    pipeline_run = models.ForeignKey(PipelineRun, verbose_name='group')
    task_type = models.IntegerField(choices=PipelineTaskType.choices())
    workload = models.ForeignKey(Workload)
    data = models.TextField()
    creation_time = models.DateTimeField()

    class Meta:  # pylint: disable=no-init
        unique_together = ("pipeline_run", "task_type", "workload")


class BackupData(BaseModel):
    result = models.ForeignKey(Result)
    raw_knobs = models.TextField()
    raw_initial_metrics = models.TextField()
    raw_final_metrics = models.TextField()
    raw_summary = models.TextField()
    knob_log = models.TextField()
    metric_log = models.TextField()
