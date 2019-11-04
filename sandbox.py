import json
import schemaValidator

schemaFile = '/Users/semeonbalagula/work/glidingdeer/scripts/schemas/bi_events.json'
dataFile = '/Users/semeonbalagula/work/glidingdeer/tmp/biEvents.json'
with open(schemaFile) as schema_data:
    jsonSchema = json.load(schema_data)

with open(dataFile) as myfile:
    for data in myfile:
        print(data)
        vldt = schemaValidator.Validate(jsonSchema, data)
        x = vldt._validateLine()
        print(x)
