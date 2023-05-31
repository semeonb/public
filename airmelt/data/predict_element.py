def merge_dict(dict1, dict2):
    return dict2.update(dict1)


class Element(object):
    def __init__(self, element_properties):
        self.element_properties = element_properties


class StaticInput(object):
    def __init__(self, data: dict):
        self.data = data


class ModelInput(object):
    """
    The input consits of a variable list and their default values
    """

    def __init__(self, structure: dict):
        self.structure = structure

    def build(self, static_input: StaticInput, element: Element):
        data = list()
        input = merge_dict(static_input, element)
        for i, v in self.structure.items():
            data.append(input.get(i, v))
        return data


class Model(object):
    def __init__(self, model_obj, input_structure):
        self.model_obj = model_obj
        self.input_structure = input_structure

    def predict(self, static_input, element_properties):
        mi = ModelInput(self.input_structure)
        data = mi.build(static_input, element_properties)
        return self.model_obj.predict(data)
