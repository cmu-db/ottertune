from celery.task.schedules import crontab
from celery.decorators import periodic_task
from celery.utils.log import get_task_logger




## periodic task example 
logger = get_task_logger(__name__)

#every 10 seconds
@periodic_task(run_every=10,name="periodic_task_example")

#every 1 min 
#@periodic_task(run_every=(crontab(hour=0,minute=1,day_of_week=0)),name="periodic_task_example")
def periodic_task_example():
    logger.info("This is a periodic task example. CHANGE ME !! " )

