import json

"""
Author: Semeon Balagula
The script receives Google BigQuery schema in json format and ndjson data line.
Usage:
import json
import schemaValidator
schemaFile = [path to your schema file]
dataFile = [path to your data file]
with open(schemaFile) as schema_data:
    jsonSchema = json.load(schema_data)
with open(dataFile) as myfile:
    for data in myfile:
        vldt = schemaValidator.ValidateRow(jsonSchema, data)
        x = vldt.validate()
        print(x)
"""


class ValidateRow(object):

    def __init__(self, jsonSchema, data):

        self.jsonSchema = jsonSchema
        self.trans_schema = self._jsonSchemaTransform()
        self.data = json.loads(data)
        self.transData = {}
        self.bq_string = 'STRING'
        self.bq_record = 'RECORD'
        self.bq_integer = 'INTEGER'
        self.bq_float = 'FLOAT'
        self.bq_bool = 'BOOLEAN'
        self.bq_nullable = 'NULLABLE'
        self.bq_repeated = 'REPEATED'
        self.list = 'list'

    def _buildFields(self, item):
        fieldsDict = {}
        for f in item.get('fields', []):
            fieldsDict[f['name']] = {
                'type': f['type'],
                'mode': f['mode'],
                'fields': self._buildFields(f)
                }
        return fieldsDict

    def _jsonSchemaTransform(self):
        trans_schema = {}
        for i in self.jsonSchema:
            trans_schema[i['name']] = {
                'type': i['type'],
                'mode': i['mode'],
                'fields': self._buildFields(i)}
        return trans_schema

    def _validateInteger(self, value):
        try:
            assert int(value) == float(value)
            return int(value)
        except Exception:
            return None

    def _validateFloat(self, value):
        try:
            assert float(value)
            return float(value)
        except Exception:
            return None

    def _validateString(self, value):
        try:
            assert isinstance(value, str)
            return value
        except Exception:
            return None

    def _validateBool(self, value):
        try:
            assert isinstance(value, bool)
            return value
        except Exception:
            return None

    def _validateType(self, schema, value):
        tp = schema['type']
        if tp == self.bq_string:
            return self._validateString(value)
        elif tp == self.bq_bool:
            return self._validateBool(value)
        elif tp == self.bq_integer:
            return self._validateInteger(value)
        elif tp == self.bq_float:
            return self._validateFloat(value)
        else:
            raise Exception('Unknown data type: {}'.format(tp))

    def _validateRepeated(self, schema, value):
        return [self._validateType(schema['type'], x) for x in value]

    def validate(self, data=None, schema=None):
        transData = {}
        if not schema:
            data, schema = self.data, self.trans_schema
        for dataKey, dataValue in data.items():
            schemaValue = schema.get(dataKey)
            if schemaValue['type'] == self.bq_record:
                nestedList = []
                nestedSchema = schemaValue['fields']
                # if object is not list, convert it into one
                if isinstance(dataValue, dict):
                    dataValue = [dataValue]
                # data value is a list of dictionaries
                for d in dataValue:
                    nestedDict = {}
                    for nestedKey, nestedValue in d.items():
                        if nestedSchema.get(nestedKey):
                            nestedDict[nestedKey] = self.validate(d, nestedSchema)
                    nestedList.append(self.validate(d, nestedSchema))
                transData[dataKey] = nestedList
            else:
                if schemaValue['mode'] == self.bq_nullable:
                    transData[dataKey] = self._validateType(schemaValue, dataValue)
                elif schemaValue['mode'] == self.bq_repeated:
                    transData[dataKey] = self._validateRepeated(schemaValue, dataValue)
        return transData
