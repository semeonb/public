import os
import shutil
import logging
import json
import csv
import datetime
import string
import random
from slacker import Slacker
from configparser import SafeConfigParser, RawConfigParser
import math


def get_relative_path(path1, path2):
    """
    Get the relative path from path1 to path2.
    """
    try:
        relative_path = os.path.relpath(path2, path1)
        return relative_path
    except ValueError:
        # Raised when paths have different drives on Windows
        return None


def millify(n, precision=0):
    """
    This function takes a numerical value 'n' and an optional argument 'precision' as inputs
    and converts it to a string with a shorthand notation for large numbers in millions, billions, and trillions.
    For example, if the input number is 5000000, the output will be "5 Million" and
    if the input number is 1000000000000, the output will be "1 Trillion".

    The 'precision' parameter determines the number of decimal places to be displayed in the output.
    If the precision is not specified or set to 0, the function returns the number rounded off to the nearest thousand with no decimal places.

    Here's a step-by-step breakdown of how the function works:

    1. The function defines a list of shorthand names for large numbers in millions, billions, and trillions as "millnames".
    2. The input number 'n' is converted to a floating-point number using float(n).
    3. The function calculates the index of the shorthand name to be used for the input number.
       This is done by taking the logarithm base 10 of the absolute value of 'n',
       dividing it by 3 and taking the floor of the result.
       This gives the number of thousand units in the input number,
       which determines the index of the shorthand name in the 'millnames' list.
       The index is then capped at the maximum length of 'millnames' minus 1 to avoid an index out of range error.
    4. The function formats the output string using string formatting methods, based on the 'precision' parameter.
    5. Finally, the function returns the output string with the shorthand notation for the input number.

    """
    millnames = ["", " Thousand", " Million", " Billion", " Trillion"]
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )
    if precision == 0:
        return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])
    elif precision == 1:
        return "{:.1f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])
    elif precision == 2:
        return "{:.2f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])
    else:
        return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def docopt_bool(var, args):
    argStr = "--{}".format(var)
    if args[argStr]:
        return True
    else:
        return False


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def divide_chunks(data, chunks_cnt):
    chunked_data = list()
    l = len(data)
    chunk_size = int(len(data) / chunks_cnt) + 1

    # looping till length l
    for i in range(0, len(data), chunk_size):
        chunked_data.append(data[i : min(i + chunk_size, l)])
    return chunked_data


def generate_random_string(length):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def delete_local_file(local_file):
    logger = logging.getLogger(__name__)
    # delete local file
    if os.path.isfile(local_file):
        try:
            os.remove(local_file)
        except Exception as e:
            logger.error(f"Error deleting local file {local_file}: {e}")


def default_dict_converter(o):
    """
    Default time converter for de-serializing dictionaries
    o: any object
    """
    if isinstance(o, datetime.datetime):
        return o.strftime("%Y-%m-%d %H:%M%:%S.%f")


def parse_bool(bool_str: str):
    """
    bool_str: the string that needs to be parsed as a boolean value.
    """
    if isinstance(bool_str, str):
        if bool_str.lower() in ["true", "t", "yes", "y", "1"]:
            return True
        elif bool_str.lower() in ["false", "f", "no", "n", "0"]:
            return False
        else:
            raise ValueError(f"{bool_str} is not a valid boolean string")
    else:
        raise TypeError(f"{bool_str} is not a string")


def parse_int(int_str, default_output=None):
    """
    int_str: the string that needs to be parsed as an integer
    default_output: the default value to return if the string cannot be parsed as an integer. If not provided, it defaults to None.
    """
    if isinstance(int_str, str):
        try:
            return int(int_str)
        except Exception:
            return default_output
    elif isinstance(int_str, int):
        return int_str
    elif isinstance(int_str, float):
        return int(int_str)
    elif int_str is None:
        return default_output
    else:
        raise TypeError(f"{int_str} is not a string")


def create_local_dir(dir_path: str, restart=False):
    """
    dir_path: the path of the directory to be created
    restart: a boolean value that indicates whether the directory should be deleted and recreated if it already exists. Defaults to False.
    """
    if os.path.isdir(dir_path):
        if restart:
            deleteLocalFolder(dir_path)
            os.makedirs(dir_path)
        pass
    else:
        os.makedirs(dir_path)


def deleteLocalFolder(folder: str):
    """
    folder: the path of the folder to be deleted
    """
    logger = logging.getLogger(__name__)
    if isinstance(folder, str):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        else:
            logger.warning(f"{folder} does not exist")
    else:
        raise TypeError(f"{folder} is not a string")


def gen_queries(folder: str, params: dict):
    """
    folder: the path of the folder containing the SQL files
    params: a dictionary of parameters to be used for string formatting the SQL queries
    """
    if os.path.isdir(folder):
        queries = []
        for f in [os.path.join(folder, f) for f in os.listdir(folder)]:
            if f.endswith(".sql"):
                queries.append(open(f).read().format(**params))
        return queries
    else:
        raise FileNotFoundError(f"{folder} is not a valid folder")


def open_json_file(full_path: str):
    """
    full_path: the path of the json file to be opened
    """
    if full_path.endswith(".json"):
        with open(full_path) as fl:
            return json.loads(fl.read())
    else:
        raise ValueError(f"{full_path} is not a json file")


def save_json_file(data, full_path):
    """
    data: a list of json objects to be written to the file
    full_path: the path of the file to which the json objects will be written
    """
    with open(full_path, "w", encoding="utf-8") as output_file:
        for dic in data:
            json.dump(dic, output_file)
            output_file.write("\n")


def csv_to_ndjson(csv_file_path, ndjson_file_path):
    # Read CSV and convert to NDJSON
    with open(csv_file_path, "r", newline="") as csvfile, open(
        ndjson_file_path, "w"
    ) as ndjsonfile:
        # Assuming the first row contains the headers
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            # Convert each row to JSON format
            json_row = json.dumps(row)

            # Write JSON row to the NDJSON file
            ndjsonfile.write(json_row + "\n")


def open_txt_file(full_path):
    """
    full_path: the path of the text file to be opened
    """
    data = open(full_path, "r").read()
    return list([i for i in data.rstrip().split(",")])


def read_config_file(config_filename: str, config_files_dir: str, safe=True):
    """
    config_filename: the name of the configuration file to be read
    config_files_dir: the directory where the configuration file is located
    safe (defaults to True): a boolean value that determines whether to use a SafeConfigParser or RawConfigParser object to read the file
    """
    full_path = os.path.join(config_files_dir, config_filename)
    if os.path.exists(full_path):
        if safe:
            parser = SafeConfigParser()
        else:
            parser = RawConfigParser()
        parser.read(full_path)
        return parser
    else:
        raise ValueError(f"{config_filename} does not exist in {config_files_dir}")


class InitLogger(object):
    def __init__(self, logger_name, filemode="w", log_dir="/tmp"):
        logger_filename = os.path.join(log_dir, "{}.log".format(logger_name))
        logging.basicConfig(
            filename=os.path.expandvars(logger_filename),
            filemode=filemode,
            level=logging.INFO,
        )


def output_builder(
    predicted_results,
    target_column_values,
    keys,
    model_name,
    submodel_name,
    execution_id,
    submodel_rank,
):
    output = list()
    for key, result in enumerate(predicted_results):
        if not target_column_values[key]:
            target_value = 0
        else:
            target_value = target_column_values[key]
        output.append(
            {
                "ts": str(execution_id),
                "target_column_current_value": str(target_value),
                "model_value": result,
                "model_name": model_name,
                "submodel_name": submodel_name,
                "id": str(keys[key]),
                "submodel_rank": str(submodel_rank),
            }
        )
    return output


def open_system_config(prod=False, config_type="SYSTEM_CONFIG"):
    # description: opens the system config file and returns the dev or prod config
    if not os.environ.get(config_type):
        raise Exception("{} env variable was not found!".format(config_type))

    try:
        config_file = open_json_file(os.environ.get(config_type))
        if prod:
            return config_file["prod"]
        else:
            return config_file["dev"]
    except Exception as ex:
        print("System config file was not found!")
        raise


class SlackBot(object):
    def __init__(self, **kwargs):
        self.slack = Slacker(token=kwargs.get("slack_token"))
        self.user = kwargs.get("slack_user")
        self.logger = logging.getLogger(__name__)

    def send_message(self, channel="general", msg=""):
        if not channel.startswith("#"):
            channel = "#" + channel
        return self.slack.chat.post_message(channel, text=msg, as_user=self.user)


class NotificationCenter(object):
    """
    The class receieves another class, representing notification platform as an input
    """

    platforms = {"slack": SlackBot}

    def __init__(self, prod=False, **kwargs) -> None:
        self.notification_platform = kwargs.get("notification_platform")
        if self.notification_platform:
            print("Notification platform: {}".format(self.notification_platform))
        self.alert_channel = kwargs.get("alert_channel")
        if self.alert_channel:
            print("Alert channel: {}".format(self.alert_channel))
        self.info_channel = kwargs.get("info_channel")
        if self.info_channel:
            print("Info channel: {}".format(self.info_channel))
        self.notification_dict = kwargs.get("notification_dict")
        if self.notification_dict:
            print("Notification dict: {}".format(self.notification_dict))
            self.info_channel, self.alert_channel = self.get_slack_channels(
                self.notification_dict, prod
            )

        try:
            self.platform_class = self.platforms[self.notification_platform]
            self.platform = self.platform_class(**kwargs)
        except Exception as ex:
            print(ex)
            print(
                "Allowable platforms for notifications are: {}".format(
                    self.platforms.keys()
                )
            )

    def send_info(self, message):
        if self.info_channel:
            return self.platform.send_message(msg=message, channel=self.info_channel)

    def send_alert(self, message):
        if self.alert_channel:
            return self.platform.send_message(msg=message, channel=self.alert_channel)

    def get_slack_channels(self, slack_dict: dict, prod: bool = False):
        if prod:
            return slack_dict["log"]["prod"], slack_dict["alert"]["prod"]
        else:
            return slack_dict["log"]["dev"], slack_dict["alert"]["dev"]
