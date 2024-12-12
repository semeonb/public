import os
from dbt.cli.main import dbtRunner, dbtRunnerResult
import datetime
from logging import getLogger

from system import toolbox


class DbtExec(dbtRunner):
    """
    The class is a wrapper for dbtRunner class.
    It is used to run dbt commands from python code.
    """

    def __init__(
        self,
        dbt_project_dir,
        model: str = None,
        prod=False,
        home_dir: str = None,
        models_dir: str = "models",
        with_prerequisites=False,
        submodel: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        """
        dbt_project_dir: a RELATIVE path to dbt project directory. For example, "dbt/my_project"
        prod: if True, the target is "prod", otherwise "dev"
        """
        self.logger = getLogger(__name__)
        self.home_dir = home_dir
        if home_dir:
            self.dbt_project_dir = toolbox.get_relative_path(home_dir, dbt_project_dir)
        else:
            self.dbt_project_dir = dbt_project_dir
        #  a RELATIVE path (to the dbt project) to a model directory. For example, "models/applovin"
        self.rel_dbt_model_dir = toolbox.get_relative_path(
            dbt_project_dir, os.path.join(dbt_project_dir, models_dir, model)
        )
        if submodel:
            self.rel_dbt_model_dir = self.rel_dbt_model_dir + "/" + submodel + ".sql"
        # if with_prerequisites is True, the model will be run with all its prerequisites
        if with_prerequisites:
            self.rel_dbt_model_dir = "+" + self.rel_dbt_model_dir
        # if prod is True, the target is "prod", otherwise "dev"
        if prod:
            self.target = "prod"
        else:
            self.target = "dev"
        # the list of dbt commands to be executed
        self.invoke_list = [
            "run",
            "--project-dir",
            self.dbt_project_dir,
            "--target",
            self.target,
            "--models",
            self.rel_dbt_model_dir,
        ]

    def run(self, vars: dict = None, *args, **kwargs):
        results = []
        if vars:
            self.invoke_list.extend(["--vars", str(vars)])
        self.logger.info(f"dbt invoke list: {self.invoke_list}")
        res: dbtRunnerResult = self.invoke(self.invoke_list)
        for r in res.result:
            results.append([f"{r.node.name}", f"{r.status}"])
            self.logger.info(f"{r.node.name}: {r.status}")
            print(f"{r.node.name}: {r.status}")
        return results


class DbtDateVars(object):
    """
    This class creates a dictionary of dbt variables from a set of table names. The output is a dictionary of a form:
    {table_name}_date_from: {date_from}, {table_name}_date_to: {date_to}

    The input is a list of tuples of a form:
    [(table_name, date_from, date_to), ...]
    """

    def __init__(
        self,
        date_from: datetime.datetime = None,
        date_to: datetime.datetime = None,
        days_offset=0,
    ):
        self.logger = getLogger(__name__)
        self.date_from = date_from
        self.date_to = date_to
        self.days_offset = days_offset
        self.vars = dict()

    def build(self, data):
        for i in data:
            model_name = i[0]
            date_from_key = "{module}_date_from".format(module=model_name)
            if not self.date_from:
                # if date_from is not specified, use the last_update_ts from the etl_objects table
                date_from_value: datetime.datetime = i[1] - datetime.timedelta(
                    days=self.days_offset
                )
            else:
                # if date_from is specified, use it
                date_from_value = datetime.datetime.strptime(self.date_from, "%Y-%m-%d")
            date_to_key = "{module}_date_to".format(module=model_name)
            if not self.date_to:
                # if date_to is not specified, use the date_to from the etl_objects table
                date_to_value: datetime.datetime = i[2]
            else:
                date_to_value = datetime.datetime.strptime(self.date_to, "%Y-%m-%d")
            self.vars[date_from_key] = date_from_value.strftime("%Y-%m-%d")
            self.vars[date_to_key] = date_to_value.strftime("%Y-%m-%d")

        self.logger.info("dbt vars: {}".format(vars))
