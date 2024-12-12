#!/usr/bin/env python3

"""
customerWindowAgg.py
Author: Semeon Balagula

The script creates statistical running window table for churn analysis


Usage:
    customerWindowAgg.py [--help] [--dateFrom <date>] [--dateTo <date>]
                         [--days_back <value>] [--churn_period <value>]
                         [--stats_lookback <value>] [--samplePoints <value>]
                         [--prod] [--notification_platform <name>]

Options:
    --dateFrom <date>               Date to run the aggregation from
    --dateTo <date>                 Date to run the aggregation to
    --days_back <value>             Running window period [default: 7]
    --churn_period <value>          Period that defines customer as churnes [default: 7]
    --stats_lookback <value>        Number of days to look back at the data, globally [default: 365]
    --samplePoints <value>          Number of sample points to take [default: 20]
    --prod                          If specified then the script will run in production mode
    --notification_platform <name>  Name of the plarform ro send notifications to [default: slack]
"""

from os import environ
import datetime
import logging
import random
from docopt import docopt

from core import gcp
from system import toolbox
from queries import customerWindowAgg

args = docopt(__doc__)

if not environ.get("SYSTEM_CONFIG"):
    raise Exception("SYSTEM_CONFIG env variable was not found!")

try:
    config_file = toolbox.open_json_file(environ.get("SYSTEM_CONFIG"))
    if args["--prod"]:
        SYSTEM_CONFIG = config_file["prod"]
    else:
        SYSTEM_CONFIG = config_file["dev"]
except Exception as ex:
    print("System config file was not found!")
    raise

BQ_PROJECT_ID = SYSTEM_CONFIG.get("bq_project_id")
GS_PROJECT_ID = SYSTEM_CONFIG.get("gs_project_id")
SQL_FILE = "customerWindowAgg.sql"
LAST_DATA_DATE_QUERY = """
    WITH data AS
    (
      SELECT
      MAX( statshour) AS lastDate
      FROM `warehouse.hourlystats`
      WHERE localStatshour >= TIMESTAMP_SUB(CURRENT_TIMESTAMP, INTERVAL 14 DAY)
    )
    SELECT MIN(lastDate) FROM data
    """
DELETE_QUERY = """
    DELETE FROM `warehouse.customerWindowAgg`
    WHERE 
    days_back={days_back} AND
    churn_period={churn_period}
    """

LOG_SLACK_CHANNEL = "bi_pipelines_log"
ALERTS_SLACK_CHANNEL = "bi_alerts"


def generateSingleDate(statsdate, days_back, churn_period):
    queryParams = {
        "statsdate": statsdate,
        "days_back": days_back,
        "churn_period": churn_period,
    }
    return customerWindowAgg.QUERY.format(**queryParams)


def loadSingleDay(statsdate, days_back, churn_period):
    logger = logging.getLogger(__name__)
    bq = gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID)
    logger.info("Started loading data for {}".format(statsdate))
    try:
        query = generateSingleDate(
            statsdate, days_back=days_back, churn_period=churn_period
        )
        logger.info("Created query for date {dt}".format(dt=statsdate))
    except Exception as e:
        msg = "Failed creating query for {dt}: {ex}".format(dt=statsdate, ex=e)
        logger.error(msg)
        raise
    try:
        bq.query2table(
            query=query, dest_table="customerWindowAgg", dest_dataset="warehouse"
        )
    except Exception as ex:
        msg = "Error loading to customer window aggregation for {dt}: {ex}".format(
            dt=statsdate, ex=ex
        )
        logger.error(msg)
        raise
    msg = "Finished loading data for {}".format(statsdate)
    logger.info(msg)


def main():
    churn_period = int(args["--churn_period"])
    days_back = int(args["--days_back"])
    stats_lookback = int(args["--stats_lookback"])
    sample_ponts = int(args["--samplePoints"])
    if args["--prod"]:
        prod = True
    else:
        prod = False

    bq = gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=BQ_PROJECT_ID)
    module_name = "customerWindowAgg_{lb}_{ch}".format(lb=days_back, ch=churn_period)
    toolbox.InitLogger(module_name, filemode="w")
    logger = logging.getLogger(__name__)

    system_config = toolbox.open_system_config(prod=prod)
    nc = toolbox.NotificationCenter(
        alert_channel=ALERTS_SLACK_CHANNEL,
        info_channel=LOG_SLACK_CHANNEL,
        notification_platform=args["--notification_platform"],
        slack_user=system_config.get("slack_user"),
        slack_token=system_config.get("slack_token"),
    )
    module_name = "customerWindowAgg_{lb}_{ch}".format(lb=days_back, ch=churn_period)
    msg = "Started loading the window data at {}".format(datetime.datetime.utcnow())
    nc.send_info(msg)
    logger.info(msg)
    bq = gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID)
    if args["--dateFrom"] and args["--dateTo"]:
        dateFrom = datetime.datetime.strptime(args["--dateFrom"], "%Y-%m-%d")
        dateTo = datetime.datetime.strptime(args["--dateTo"], "%Y-%m-%d")
    else:
        try:
            lastDate = bq.query2vector(query=LAST_DATA_DATE_QUERY)[0]
            logger.info("last date is {}".format(lastDate))
            dateTo = lastDate - datetime.timedelta(days=churn_period)
        except Exception as ex:
            msg = "Could not fetch last date: {}".format(str(ex))
            logger.error(msg)
            nc.send_alert(msg)
            raise
        try:
            dateFrom = lastDate - datetime.timedelta(days=stats_lookback)
            logger.info("last date is {}".format(dateFrom))
        except Exception as ex:
            logger.info(ex)
            dateFrom = datetime.datetime(2019, 1, 1).date()
    logger.info(
        "Starting loading data from {dfrom} to {dtto}".format(
            dfrom=dateFrom, dtto=dateTo
        )
    )
    try:
        bq.execute_query(
            query=DELETE_QUERY.format(days_back=days_back, churn_period=churn_period)
        )
    except Exception as ex:
        msg = "could not delete from table customerWindowAgg:{}".format(ex)
        logger.warning(msg)
    statsdate = dateFrom
    diff = dateTo - dateFrom
    sample_days = random.sample(list(range(diff.days)), sample_ponts)
    date_points = [
        (dateFrom + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in sample_days
    ]

    for statsdate in date_points:
        loadSingleDay(statsdate, days_back, churn_period)
        print(
            "Loaded date {dt}; days_back: {db}; churn_period: {cp}".format(
                dt=statsdate, db=days_back, cp=churn_period
            )
        )


if __name__ == "__main__":
    main()
