import sys
import numpy as np 
import time 
from util import Data
from typing import ( List, Dict, Any, Type, TypeVar, Union ) 

ARRAY = TypeVar('numpy.ndarray')


def print_metrics(text: List[str], correct: int, start: float, i: int):

    elapsed_time = float(time.time() - start)
    text_per_second = i / elapsed_time if elapsed_time > 0 else 0

    sys.stdout.write(f"\r READ: {str(100 * i/float(len(text)))[:4]} \
                        SPEED: (text/sec): {str(text_per_second)[0:5]} \
                        Correct: {str(correct)} \
                        Trained: {str(i+1)} \
                        Training Accuracy: {str(correct * 100 / float(i+1))[:4]} %" )


class sentiNet:

    def __init__(text: List[str], labels: List[str], reduce_noise: bool, hidden_nodes: int, 
                 min_count: int = 10, polarity_cutoff: float = 0.1, learning_rate: float = 0.1):

        self.DATA = Data(text=text, labels=labels, 
                         polarity_cutoff=polarity_cutoff,
                         min_count=min_count, reduce_noise=reduce_noise)


    def init_network(self, input_nodes: int, hidden_nodes: int, 
                           output_nodes: int, learning_rate: float = 1.0):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate


        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                            (self.hidden_nodes, self.output_nodes))
        
        self.layer_1 = np.zeros((1, self.hidden_nodes))


    def convert_label_to_target(self, label: str) -> int:

        if label == 'severe':

            return 1

        else:

            return 0

    def sigmoid(self, x: float) -> float:

        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output: float) -> float:

        return output * ( 1 - output )

    def train(self, text_raw: List[str], labels: List[str]):

        training_text = list()

        for text in text_raw:

            indices = set()

            for word in text.split(" "):

                if word in self.DATA.w2i.keys():

                    indices.add(self.DATA.w2i[word])

            training_text.append(list(indices))

        assert(len(training_text) == len(labels))

        correct = 0

        start = time.time()

        for i in range(len(training_text)):

            text = training_text[i]
            label = labels[i]

            ###FORWARD PASS###

            self.layer_1 *= 0

            for index in text:

                self.layer_1 += self.weights_0_1[index]

            layer_2 = self.sigmoid(self.layer_1.dot(self.weight_1_2))

            ##BACKWARD PASS###

            layer_2_loss = layer_2 - self.convert_label_to_target(label)
            layer_2_loss = layer_2_loss * self.sigmoid_output_2_derivative(layer_2)

            layer_1_loss = layer_2_loss.dot(self.weights_1_2.T)
            
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_loss) * self.learning_rate

            ###UPDATE WEIGHTS###

            for index in text:

                self.weights_0_1[index] -= layer_1_loss[0] * self.learning_rate

            if layer_2 >= 0.5 and label == 'severe':
                correct += 1
            
            elif layer_2 < 0.5 and label == 'nonsevere':
                correct += 1

            print_metrics(text=training_text, correct=correct, 
                          start=start, i=i)

            if i % 1000 == 0:
                print("")
    
    def predict(self, test_text: List[str], test_labels: List[str]) -> Dict[str, str]:

        correct = 0

        start = time.time()

        preds = []

        for i in range(len(test_text)):

            pred = self.classify(test_text[i])
            preds.append(pred)

            if pred == test_labels[i]:

                correct += 1

            print_metrics(text=test_text, correct=correct, 
                          start=start, i=i)

        return dict(zip(preds, test_labels))

    def classify(self, text: str) -> str:

        self.layer_1 *= 0

        unique_indicies = set()

        for word in text.lower().split(" "):

            if word in self.DATA.w2i.keys():

                unique_indicies.add(self.DATA.w2i[word])

        for index in unique_indicies:

            self.layer_1 += self.weights_0_1[index]

        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if layer_2[0] >= 0.5:

            return 'severe'

        else:

            return 'nonsevere'