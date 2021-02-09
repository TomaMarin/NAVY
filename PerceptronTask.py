import xml.dom.minidom
from xml.dom.minidom import Node


def getChildrenByTagName(node, tagName):
    for child in node.childNodes:
        if child.nodeType == child.ELEMENT_NODE and (tagName == '*' or child.tagName == tagName):
            yield child


class InputDescription:
    def __init__(self, min, max, name):
        self.min = min
        self.max = max
        self.name = name

    def __repr__(self):
        return "InputDescription /" + str(self.name) + "\ with min val " + str(self.min) + " and max val " + str(
            self.max)


class TrainTaskElement:
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

    def __repr__(self):
        return "TrainTaskElement with inputs on " + str(self.inputs) + " and output val " + str(self.output)


class TestTaskElement:
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None

    def __repr__(self):
        return "TestTaskElement with inputs on " + str(self.inputs)+ " and output val"+ str(self.output)


class PerceptronTask:
    def __init__(self):
        self.testElements = list()
        self.trainElements = list()
        self.min_max_vals_desc = list()

    def parse_xml(self, file_name, root_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.getElementsByTagName(root_name)
        testElements = list()
        trainElements = list()
        min_max_vals_desc = list()
        for node in root:
            # read input desc
            for i, desc in enumerate(node.getElementsByTagName('inputDescriptions')):
                self.min_max_vals_desc.append(
                    InputDescription(desc.getElementsByTagName("minimum")[0].firstChild.nodeValue,
                                     desc.getElementsByTagName("maximum")[0].firstChild.nodeValue,
                                     desc.getElementsByTagName("name")[0].firstChild.nodeValue))

            # read test set
            for elements in node.getElementsByTagName('TestSet')[0].getElementsByTagName('element'):
                for inputs in elements.getElementsByTagName('inputs'):
                    vals = list()
                    for values in inputs.getElementsByTagName('value'):
                        vals.append(float(values.firstChild.nodeValue))
                    p = TestTaskElement(vals)
                    self.testElements.append(p)
            # read train set
            for elements in node.getElementsByTagName('TrainSet'):
                for iterator in range(elements.getElementsByTagName('element').length):
                    out_val = float(elements.getElementsByTagName('output')[iterator].firstChild.nodeValue)
                    attribute_vals = list()
                    for vals in elements.getElementsByTagName('inputs')[iterator].getElementsByTagName('value'):
                        attribute_vals.append(float(vals.firstChild.nodeValue))
                    p = TrainTaskElement(attribute_vals, out_val)
                    self.trainElements.append(p)
