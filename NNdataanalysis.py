from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import unittest
import random
from collections import deque
import math
import matplotlib.pyplot as plt


class DLLNode:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    class EmptyListError(Exception):
        pass

    def __init__(self):
        """
        _head: Reference to the first node in the DLL
        _tail: Reference to the last node in the DLL
        _curr: Reference to the current node in the DLL
        """
        self._head = None
        self._tail = None
        self._curr = None

    def move_forward(self):
        """ Move the current pointer forward to the next node and return
        data. """
        # if list is empty
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        # if not, set the current node to the next one
        # if the next one is none, then you're already at a tail node and
        # you can't move forward
        if self._curr.next is None:
            raise IndexError
        else:
            self._curr = self._curr.next
        return self._curr.data

    def move_back(self):
        """ Move the current pointer backward to the previous node and return
        data. """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        # if current node is at the head, return indexerror
        if self._curr.prev is None:
            raise IndexError
        else:
            self._curr = self._curr.prev
        return self._curr.data

    def reset_to_head(self):
        """ Reset current pointer to the DLL head."""
        self._curr = self._head
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def reset_to_tail(self):
        """ Reset current pointer to the DLL tail."""
        self._curr = self._tail
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def add_to_head(self, data):
        """ Create a new DLL node and add to the head."""
        new_node = DLLNode(data)
        if self._head is None:
            self._head = new_node
            self._tail = new_node
        else:
            # link new node to DLL
            new_node.next = self._head
            self._head.prev = new_node
            # set the DLL head to the new node
            self._head = new_node
        # set the current pointer to the head (aka new node)
        self.reset_to_head()
        # node_previous remains as none

    def add_after_cur(self, data):
        """ Create a new DLL node and link it to the current node."""
        # if the DLL is empty, the new node becomes the new head
        if self._curr is None:
            self.add_to_head(data)
            return
        # otherwise create a node here
        new_node = DLLNode(data)
        # set new node's pointer to the current node's next
        new_node.next = self._curr.next
        # but if the DLL has only one node, then we set tail to be new node
        # and then we need to link the new node to the
        if new_node.next is None:
            self._tail = new_node
            self._tail.prev = self._curr
        else:
            self._curr.next.prev = new_node
        # link the new node to the prvious
        new_node.prev = self._curr
        # link the current node to the new downstream node
        self._curr.next = new_node

    def remove_from_head(self):
        """ Remove a new DLL node and return value"""
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        """ Remove a node after the current node and return value"""
        # if list is empty, raise empty list error
        if self._head is None or self._curr is None:
            raise DoublyLinkedList.EmptyListError
        if self._curr.next is None:
            raise IndexError
        ret_val = self._curr.next.data
        self._curr.next = self._curr.next.next
        if self._curr.next is None:
            self._tail = self._curr
        else:
            self._curr.next.prev = self._curr
        return ret_val

    def get_current_data(self):
        if self._head is None or self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    # iterator
    def __iter__(self):
        # make sure that the iterator starts at the beginning of the DLL
        self._curr_iter = self._head
        return self

    def __next__(self):
        """Return value in current node and move to next item in the list"""
        # if current iterator is none, there is no data.
        if self._curr_iter is None:
            raise StopIteration
        # capture the data in current node
        ret_val = self._curr_iter.data
        self._curr_iter = self._curr_iter.next
        return ret_val


class DataMismatchError(Exception):
    """ Exception raised when data does not have as many labels as
    it does features.
    """
    pass


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class NNData:
    """ Class containing methods to efficiently manage our testing and
    training data.
    """

    class Order(Enum):
        """ Define whether the training data is presented in the same
        order to the neural network each time or in random order.
        """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """ Identify whether we are requesting training set or testing
        set data.
        """
        TRAIN = 0
        TEST = 1

    def __init__(self, features: list = None, labels: list = None,
                 train_factor: float = 0.9):
        """
        Instance attributes:
            _features: list of lists containing features in a NNdataset
            _labels: list of lists containing labels in a NNdataset
            _train_factor: float indicating training factor
            _train_indices: indirect ref list to point to items in
                            training dataset
            _test_indices: indirect ref list to point to items in
                           testing dataset
            _train_pool: deque to keep track of which training items
                        that have not been seen in a training epoch
            _test_pool: deque to keep track of which training items have
                        not been seen in a testing run
        """
        # create empty lists if features/labels defaults to None
        if features is None:
            features = []
        if labels is None:
            labels = []

            # set the instance attributes to default
        self._features = None
        self._labels = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()

        # initialize internal data
        self._train_factor = NNData.percentage_limiter(train_factor)
        self.load_data(features, labels)
        self.split_set()
        self.prime_data()

    @staticmethod
    def percentage_limiter(percentage: float):
        """ Contain training factor to valid percentage.
        """
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        else:
            return percentage

    def load_data(self, features: list = None, labels: list = None):
        """ Verify labels and features passed in and store as numpy arrays.
        Arg:
            features: list of lists containing features in a NNdataset
            labels: list of lists containing labels in a NNdataset
        """
        # if features is None, assume client wants to clear the data
        if features is None:
            self._features = None
            self._labels = None
            return
        # compare the length of features and labels
        elif len(features) != len(labels):
            raise DataMismatchError
        else:
            # create numpy arrays from features and labels and assign
            try:
                self._features = np.array(features, dtype=float)
                self._labels = np.array(labels, dtype=float)
            except ValueError:
                self._features = None
                self._labels = None

    def split_set(self, new_train_factor: float = None):
        """ Split a dataset into training and testing sets by creating
        respective indirect reference lists.
        Arg:
            new_train_factor: float representing new training factor
            passed by user
        """
        # if a new train factor was passed, calculate the new train factor
        if new_train_factor is not None:
            self._train_factor = \
                NNData.percentage_limiter(new_train_factor)
            # reset lists
            self._train_indices = []
            self._test_indices = []

        if self._features is None:
            self._train_indices = []
            self._test_indices = []
            return
        else:
            len_examples = len(self._features)
            len_trainers = round(self._train_factor * len_examples)

            # set up _train_indices
            while len(self._train_indices) < len_trainers:
                # generate a random int between 0 and number of examples
                index = random.randint(0, (len_examples - 1))
                # if the index is already in the list, skip. if not, add.
                if index not in self._train_indices:
                    self._train_indices.append(index)
                else:
                    continue

            # set up _test_indices
            while len(self._test_indices) < \
                    (len_examples - len(self._train_indices)):
                index = random.randint(0, (len_examples - 1))
                if index not in self._train_indices and \
                        index not in self._test_indices:
                    self._test_indices.append(index)
                else:
                    continue

        # sort the lists
        self._train_indices.sort()
        self._test_indices.sort()

    def prime_data(self, target_set=None, order=None):
        """ Load one or both deques to be used as indirect indices. Shuffle
        the deques if specified by user. The test and train indexes are used
        to form the deques.
        """
        # clear deque if its already populated
        self._train_pool = deque()
        self._test_pool = deque()

        # if no target_set is specified, load both deques
        if target_set is None:
            self._train_pool.extend(self._train_indices)
            self._test_pool.extend(self._test_indices)
        elif target_set == NNData.Set.TRAIN:
            self._train_pool.extend(self._train_indices)
        else:
            self._test_pool.extend(self._test_indices)

        if order is NNData.Order.RANDOM:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)
        elif order is NNData.Order.SEQUENTIAL or None:
            pass

    def get_one_item(self, target_set=None):
        """ Return a feature/label pair from specified data set by popping from
        the specified deque.
        Arg:
            target_set: Enum representing dataset to pull from.
        Return:
            Tuple representing one feature/label pair.
        """
        # determine target set
        if target_set is NNData.Set.TRAIN:
            # make shallow copy deque
            active_pool = self._train_pool
        elif target_set is None:
            active_pool = self._train_pool
        else:
            active_pool = self._test_pool

        try:
            # select index to use
            index = active_pool[0]
            # remove index from deque so that it isn't reused
            active_pool.popleft()
            # return tuple
            x = self._features[index], self._labels[index]
            return x
            # if no indices left, return None
        except IndexError:
            return None

    def number_of_samples(self, target_set=None):
        """ Return total number of examples in a dataset.
        Arg:
            target_set: Enum indicating dataset to determine length of.
        Return:
            Total number of testing and/or training examples
        """
        if target_set is NNData.Set.TEST:
            # return total number of testing examples
            return len(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            # return total number of training examples
            return len(self._train_indices)
        elif target_set is None:
            # return both combined
            return len(self._test_indices) + len(self._train_indices)

    def pool_is_empty(self, target_set=None):
        """ Check if targeted pool is empty.
        Return:
            True if pool is empty. False if pool is not empty.
        """
        if target_set is NNData.Set.TEST:
            if self._test_pool:
                return False
            else:
                return True
        elif target_set is NNData.Set.TRAIN:
            if self._train_pool:
                return False
            else:
                return True
        else:
            if self._train_pool:
                return False
            else:
                return True


class MultiLinkNode(ABC):
    class Side(Enum):
        """ Enum subclass to identify relationships between neurodes.
        """
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        """
        Instance attributes:
            _reporting_nodes: Dict representing nodes directly upstream or
                             downstream of active node with information to
                             report. Keys = SideEnum. Values = binary encoding
                             to keep track of which neighboring nodes have
                             info avail.
            _reference_value: Dictionary representing binary encoded values
                             that indicate when all neighboring nodes have been
                             reported. Keys = SideEnum. Values = binary
                             encoding representing value (ref value) when all
                             of the nodes have been reported.
            _neighbors: Dictionary with two entries. Keys = SideEnum. Values =
                        references to neighboring nodes upstream and
                        downstream.
        """
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        """ Return ID of the node and the ID's of the neighboring nodes
        upstream and downstream.
        """
        upstream_neighbors = [id(node) for node in
                              self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        downstream_neighbors = [id(node) for node in
                                self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]
        return f"Node ID is {id(self)}. " \
               f"Upstream nodes are: {upstream_neighbors} " \
               f"Downstream nodes are: {downstream_neighbors} "

    @abstractmethod
    def _process_new_neighbor(self, node, side: Side):
        pass

    def reset_neighbors(self, node: list, side: Side):
        """ Populate _neighbors with list of neighboring nodes. Calculate
        corresponding _reference value.
        Arg:
            node: list of references to the neighboring nodes
            side: Side enum indicating upstream or downstream
        """
        # shallow copy list of nodes into the appropriate key of neighbors dict
        if side in (MultiLinkNode.Side.UPSTREAM,
                    MultiLinkNode.Side.DOWNSTREAM):
            self._neighbors[side] = node.copy()
            # call _process_new_neighbor() for each node
            for i in range(len(self._neighbors[side])):
                self._process_new_neighbor(self._neighbors[side][i], side)

        # update the reference values with all nodes reported
        self._reference_value[side] = 2 ** (len(node)) - 1


class Neurode(MultiLinkNode):
    def __init__(self, node_type, learning_rate=.05):
        """
        Instance Attributes:
            _value: current sigmoid value of the neurode.
            _node_type: one of the LayerType elements and represents the
            role of this neurode.
            _learning_rate: learning rate used in backpropogation.
            _weights: Dict representing the weights given to the upstream
            connections. Keys = references to upstream neurodes. Values =
            floats representing the weights.
        """
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    def _process_new_neighbor(self, node, side):
        """ When an upstream neighbor is added, add the node reference as a
        key to the self._weights. Value to this key would be a randomly gen.
        float between 1 and 0
        """
        if side == Neurode.Side.UPSTREAM:
            self._weights[node] = random.random()
        else:
            pass

    def _check_in(self, node, side):
        """ Report a neighboring node and check if all nodes have reported.
        Arg:
            node: Reference to neighboring node that is reported
            side: Enum class indicating whether the node is upstream
        Return:
            Bool indicating whether all nodes have been reported or not
        """
        # find the node's index in _neighbors
        if side in (MultiLinkNode.Side.UPSTREAM,
                    MultiLinkNode.Side.DOWNSTREAM):
            # if node exists as a element in list value for self._neighbors
            if node in self._neighbors[side]:
                # retrieve index
                index = self._neighbors[side].index(node)
                node_report = 2 ** index
                # do the bitwise operation
                self._reporting_nodes[side] = \
                    node_report | self._reporting_nodes[side]
            if self._reporting_nodes[side] == self._reference_value[side]:
                self._reporting_nodes[side] = 0
                return True
            else:
                return False

    def get_weight(self, node):
        """ Return upstream node's weight relative to current node.
        Arg:
            node: Reference to upstream node whose weight will be returned.
        Return:
            Weight of the upstream node relative to the current node.
        """
        if node in self._weights:
            return self._weights[node]

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        if 0 < learning_rate <= 1.0:
            self._learning_rate = learning_rate
        else:
            return ValueError


class FFNeurode(Neurode):
    """ Feed forward neurode class"""

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        """ Return value of node's sigmoid function.
        """
        sigmoid_value = 1 / (1 + np.exp(-value))
        return sigmoid_value

    def _calculate_value(self):
        """ Calculate the weighted sum of the upstream nodes' values.
        """
        upstream_values = []
        upstream_weights = []
        weighted_sum = 0

        # compile a list of all the upstream values
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            upstream_values.append(node._value)
            # compile a list of all the upstream nodes
            upstream_weights.append(self._weights[node])

        for i in range(len(upstream_values)):
            weighted_sum += upstream_weights[i] * upstream_values[i]

        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """ Node sends signal to downstream nodes when it has data
        available.
        """
        # if there are downstream neighbors,
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Called by upstream nodes when they have data ready. Prompts current
        node to act on data if all nodes have reported.
        Arg:
            node: reference to upstream node that passes itself in
        """
        all_checked_in = self._check_in(node, MultiLinkNode.Side.UPSTREAM)
        if all_checked_in:
            # collect data and make it available to the next layer
            self._calculate_value()
            # trigger current node to send signal to downstream neighbors
            self._fire_downstream()

    def set_input(self, input_value):
        """ Allow client to directly set the value of an input layer neurode.
        Call data_ready_upstream on all of the downstream neighbors.
        """
        # set the value of input_layer neurode
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        """ Calculate the sigmoid derivative.
        Arg:
            value: Calculated sigmoid value at x.
        """
        return value * (1 - value)

    def _calculate_delta(self, expected_value=None):
        """ Calculate the delta of output or hidden node.
        """
        # if there are no neurodes downstream, we've hit an output neurode
        if self._node_type == LayerType.OUTPUT:
            self._delta = (expected_value - self._value) * \
                          self._sigmoid_derivative(self._value)
        else:
            # calculate weighted sum of downstream deltas
            weighted_sum = 0
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                weighted_sum += node._delta * node._weights[self]
            # calculate the delta based on weighted sum and sigmoid derivative
            self._delta = weighted_sum * self._sigmoid_derivative(self._value)

    def data_ready_downstream(self, node):
        """ Called by downstream neurodes when they have data ready.
        Arg:
            node: Reference to downstream node
        """
        all_checked_in = self._check_in(node, MultiLinkNode.Side.DOWNSTREAM)
        if all_checked_in:
            self._calculate_delta(None)
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """ Set an output neurode's value to an expected value determined
        by user.
        Arg:
            expected_value: self explanatory
        """
        self._calculate_delta(expected_value)
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        """ Adjust weights between a downstream neurode and current node.
        Arg:
            node: Reference to an upstream node
            adjustment: Int indicating the new upstream node's weight
        """
        if node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node._weights[self] += adjustment

    def _update_weights(self):
        """ Determine weight adjustment for all downstream nodes and update
        the weights accordingly.
        """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            # determine the new adjustment
            adjustment = node._delta * self._value * node._learning_rate
            # set the new weight
            self.adjust_weights(node, adjustment)

    def _fire_upstream(self):
        """ Let upstream nodes know that a downstream node has data available.
        """
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta


class FFBPNeurode(BPNeurode, FFNeurode):
    pass


class LayerList(DoublyLinkedList):
    """ Doubly linked list specialized to the neural network.
    """

    def __init__(self, inputs: int, outputs: int):
        """ Set up the DLL with the head node (input layer) and the tail node
        (output layer)
        inputs: int representing number of input layer neurodes
        outputs: int representing number of output layer neurodes
        """
        super().__init__()

        # create a list of FFBP neurodes with n=inputs elements
        inputs_layer = []
        for i in range(inputs):
            inputs_layer.append(FFBPNeurode(LayerType.INPUT))

        # create a list of FFBP neurodes with n=outputs elements
        outputs_layer = []
        for i in range(outputs):
            outputs_layer.append(FFBPNeurode(LayerType.OUTPUT))

        # set the inputs list as the head and the outputs list as the tail
        self.add_to_head(inputs_layer)
        self.add_after_cur(outputs_layer)

        # link each head/tail neurode to each other
        for neurode in self._head.data:
            neurode.reset_neighbors(self._tail.data,
                                    MultiLinkNode.Side.DOWNSTREAM)
        for neurode in self._tail.data:
            neurode.reset_neighbors(self._head.data,
                                    MultiLinkNode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        """ Add a hidden layer after the current layer to the DLL.
        A layer is a list containing n = num_nodes FFBP objects.
        """
        hidden_layer = []
        for i in range(num_nodes):
            hidden_layer.append(FFBPNeurode(LayerType.HIDDEN))

        if self._curr is self._tail:
            raise IndexError
        else:
            self.add_after_cur(hidden_layer)
        # temporarily set the new added layer as the current node
        self.move_forward()
        # link the neurodes in new layer to their upstr and downstr neighbors
        for neurode in self._curr.data:
            neurode.reset_neighbors(self._curr.next.data,
                                    MultiLinkNode.Side.DOWNSTREAM)
            neurode.reset_neighbors(self._curr.prev.data,
                                    MultiLinkNode.Side.UPSTREAM)
        # link the neurodes in previous layer to new downstream neighbors
        for neurode in self._curr.prev.data:
            neurode.reset_neighbors(self._curr.data,
                                    MultiLinkNode.Side.DOWNSTREAM)
        # link the neurodes in the downstr layer to the new upstr neighbors
        for neurode in self._curr.next.data:
            neurode.reset_neighbors(self._curr.data,
                                    MultiLinkNode.Side.UPSTREAM)
        self.reset_to_head()

    def remove_layer(self):
        """ Remove a hidden layer after the current layer from the DLL."""
        # prevent removal of non-hidden layer
        if self._curr.next is self._tail:
            raise IndexError
        else:
            # relink the neurodes in current node to the new downstr neighbors
            for neurode in self._curr.data:
                neurode.reset_neighbors(self._curr.next.next.data,
                                        MultiLinkNode.Side.DOWNSTREAM)
            # link the neurodes in the downstr layer to the new upstr neighbors
            for neurode in self._curr.next.next.data:
                neurode.reset_neighbors(self._curr.data,
                                        MultiLinkNode.Side.UPSTREAM)
            self.remove_after_cur()

    @property
    def input_nodes(self):
        """ Return a list of the input neurodes"""
        return self._head.data

    @property
    def output_nodes(self):
        """ Return a list of the output neurodes"""
        return self._tail.data


class FFBPNetwork:
    """Use the Layer list to create a network. Provide testing and training
    methods. """

    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        """ Create a LayerList class with passed inputs and outputs.
        network = Layerlist (DLL) object
        _num_inputs = total number of input nodes
        _num_outputs = total number of output nodes
        inputs = list of references to the input nodes
        outputs = list of references to the output nodes
        """
        # create a Layerlist class with passed inputs and outputs
        self.network = LayerList(num_inputs, num_outputs)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self.inputs = self.network.input_nodes
        self.outputs = self.network.output_nodes

    @staticmethod
    def print_results(input_val, expected_val, predicted_val):
        """ Generate a table comparing the results of each training test in
        an epoch."""
        results_dict = {"Input Values": input_val,
                        "Expected Values": expected_val,
                        "Predicted Values": predicted_val}
        for key in results_dict:
            print(f"{key:<20}", end="")
        print("\n", end="")
        for j in range(len(input_val)):
            for key in results_dict.keys():
                try:
                    print(f"{results_dict[key][j]:<20.2f}", end='')
                except IndexError:
                    pass
                if key == "Predicted Values":
                    print("\n", end="")

    def add_hidden_layer(self, num_nodes: int = 5, position=0):
        """ Add a hidden layer with a given number of nodes. By default,
        the hidden layer is added after the input layer.
        Arg:
            num_nodes: Number of neurodes to add to the layer
            position: Number indicating where to insert new hidden layer."""
        self.network.reset_to_head()
        for i in range(position):
            self.network.move_forward()
        self.network.add_layer(num_nodes)
        self.network.reset_to_head()

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        if data_set.pool_is_empty(NNData.Set.TRAIN):
            raise FFBPNetwork.EmptySetException
        epoch_list = []
        rmse_list = []
        for i in range(epochs):
            # reprime the data using the specified order
            data_set.prime_data(None, order)
            sum_error = 0

            # go through all the examples of a testing set
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                predicted_outputs = []
                features, target_outputs = data_set.get_one_item(
                    NNData.Set.TRAIN)
                # present jth feature to jth input neurode in input layer
                for j in range(len(features)):
                    self.inputs[j].set_input(features[j])
                # append the predicted values of each neurode to a list
                for neurode in self.outputs:
                    predicted_outputs.append(neurode.value)
                for j in range(len(target_outputs)):
                    # present the expected values to the output neurodes
                    self.outputs[j].set_expected(target_outputs[j])
                    # calculate the error
                    sum_error += (self.outputs[j].value - target_outputs[j]) \
                                 ** 2 / self._num_outputs
                # check verbosity and print out the results
                if verbosity > 1 and (i / 1000).is_integer():
                    self.print_results(features, target_outputs,
                                       predicted_outputs)

            if i % 100 == 0 and verbosity > 0:
                print("Epoch", i, "RMSE = ", math.sqrt(
                    sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

                # collect RMSE and epoch data to graph
                rmse_list.append(math.sqrt(
                    sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
                epoch_list.append(i)

        print("Final Training Epoch RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

        return epoch_list, rmse_list

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """ Test the network."""
        if data_set.pool_is_empty(NNData.Set.TEST):
            raise FFBPNetwork.EmptySetException

        data_set.prime_data(order)

        sum_error = 0

        test_x = []
        test_y = []

        while not data_set.pool_is_empty(NNData.Set.TEST):
            predicted_outputs = []
            features, target_outputs = data_set.get_one_item(NNData.Set.TEST)

            # present jth feature to jth input neurode in input layer
            for j in range(len(features)):
                self.inputs[j].set_input(features[j])

            # append the predicted values of each neurode to a list
            for neurode in self.outputs:
                predicted_outputs.append(neurode.value)
            for j in range(len(target_outputs)):
                sum_error += (self.outputs[j].value - target_outputs[j]) \
                             ** 2 / self._num_outputs
            self.print_results(features, target_outputs,
                               predicted_outputs)

            test_x.append(features)
            test_y.append(predicted_outputs)

        print("Test Epoch RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TEST)))

        return test_x, test_y


def run_iris(train_factor=0.7, num_nodes=3, epoch=10001):
    """Test and train iris dataset.
    Data source: https://www.kaggle.com/uciml/iris"""
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(num_nodes)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2],
              [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
              [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
              [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2],
              [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
              [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4],
              [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4],
              [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
              [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2],
              [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1],
              [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3],
              [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3],
              [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3],
              [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1],
              [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4],
              [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5],
              [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3],
              [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4],
              [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1],
              [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6],
              [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3],
              [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1],
              [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3],
              [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7],
              [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5],
              [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4],
              [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5],
              [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8],
              [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8],
              [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2],
              [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4],
              [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1],
              [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9],
              [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2],
              [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, train_factor)
    x, y = network.train(data, epoch, order=NNData.Order.RANDOM)
    network.test(data)
    return x, y


def run_sin(hidden_layer: int = 1, train_factor=0.1, epoch=10001):
    """Test and train sine function from 0 radians to 1.57 radians."""
    network = FFBPNetwork(1, 1)
    for i in range(hidden_layer):
        network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2],
             [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33],
             [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46],
             [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59],
             [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72],
             [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85],
             [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98],
             [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24],
             [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37],
             [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328],
             [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501],
             [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962],
             [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737],
             [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392],
             [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995],
             [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487],
             [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193],
             [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, train_factor)
    network.train(data, epoch, order=NNData.Order.RANDOM)
    if train_factor == 1:
        data.split_set(0)
        data.prime_data()
        x, y = network.test(data)
    else:
        x, y = network.test(data)
    return x, y

    # Plot the graph
    # plt.plot(sin_X, sin_Y, "or", label="Actual")
    # plt.plot(x, y, "ob", label="Predicted, tf = 0.1")
    # plt.title("Sin curve from 0 to 1.57 radians, actual vs. predicted
    # values")
    # plt.legend()
    # plt.show()


def run_xor(hidden_layer: int = 1, num_nodes=3, epoch=20001):
    network = FFBPNetwork(2, 1)
    # add a hidden layer of 3 neurodes
    for i in range(hidden_layer):
        network.add_hidden_layer(num_nodes)
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    # train on all of the data
    train_factor = 1
    data = NNData(features, labels, train_factor)
    x, y = network.train(data, epoch, order=NNData.Order.RANDOM)
    data.split_set(0)
    data.prime_data()
    network.test(data)
    return x, y


def run_xor_bias(hidden_layer: int):
    network = FFBPNetwork(3, 1)
    for i in range(hidden_layer):
        network.add_hidden_layer(3)
    features = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
    labels = [[0], [1], [1], [0]]
    # train on all of the data
    train_factor = 1
    data = NNData(features, labels, train_factor)
    x, y = network.train(data, 20001, order=NNData.Order.RANDOM)
    data.split_set(0)
    data.prime_data()
    network.test(data)
    return x, y


def sin_output():
    """Put sine curve from 0 -> 1.57 radians through NN with different training
    factors. """
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2],
             [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33],
             [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46],
             [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59],
             [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72],
             [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85],
             [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98],
             [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24],
             [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37],
             [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328],
             [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501],
             [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962],
             [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737],
             [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392],
             [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995],
             [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487],
             [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193],
             [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]

    test_x, test_y = run_sin(0.1)
    test_x2, test_y2 = run_sin(0.5)
    test_x1, test_y1 = run_sin(1)

    plt.plot(sin_X, sin_Y, "or", label="Actual")
    plt.plot(test_x, test_y, "ob", label="Predicted, tf = 0.1")
    plt.plot(test_x2, test_y2, "oy", label="Predicted, tf = 0.5")
    plt.plot(test_x1, test_y1, "og", label="Predicted, tf = 1")
    plt.title("Sin curve from 0 to 1.57 radians, actual vs. predicted values")
    plt.legend()
    plt.show()


def extra_credit_1():
    """ Train XOR dataset with and without a bias node to compare RMSE
    values over an average of 5 cycles, 20000 epochs each."""
    # create seed array
    x, y = run_xor(3)
    x1, y1 = run_xor_bias(3)

    x = np.array(x)
    sum_y = np.array(y)
    sum_y1 = np.array(y1)

    for i in range(4):
        x, y = run_xor(3)
        x1, y1 = run_xor_bias(3)

        sum_y += np.array(y)
        sum_y1 += np.array(y1)

    avg_y = sum_y / 5
    avg_y1 = sum_y1 / 5

    plt.plot(x, avg_y, "or", label="No bias node")
    plt.plot(x, avg_y1, "ob", label="Bias node")
    plt.title("XOR bias vs. non-bias")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def extra_credit_2():
    """train XOR function with 1, 3, and 5 hidden layers"""
    # create seed array
    x, y = run_xor_bias(1)
    x1, y1 = run_xor_bias(3)
    x2, y2 = run_xor_bias(5)

    x = np.array(x)
    sum_y = np.array(y)
    sum_y1 = np.array(y1)
    sum_y2 = np.array(y2)

    for i in range(4):
        x, y = run_xor_bias(1)
        x1, y1 = run_xor_bias(3)
        x2, y2 = run_xor_bias(5)

        sum_y += np.array(y)
        sum_y1 += np.array(y1)
        sum_y2 += np.array(y2)

    avg_y = sum_y / 5
    avg_y1 = sum_y1 / 5
    avg_y2 = sum_y2 / 5

    plt.plot(x, avg_y, "or", label="1")
    plt.plot(x, avg_y1, "ob", label="3")
    plt.plot(x, avg_y2, "oy", label="5")
    plt.title("XOR with different # of hidden layers")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def extra_credit_3():
    """ Train sin() function from 0 to 2pi (6.28 radians). """
    sin_X = [[0.00], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.10], [0.11], [0.12], [0.13], [0.14], [0.15],
             [0.16], [0.17], [0.18], [0.19], [0.20], [0.21], [0.22], [0.23],
             [0.24], [0.25], [0.26], [0.27], [0.28], [0.29], [0.30], [0.31],
             [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39],
             [0.40], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47],
             [0.48], [0.49], [0.50], [0.51], [0.52], [0.53], [0.54], [0.55],
             [0.56], [0.57], [0.58], [0.59], [0.60], [0.61], [0.62], [0.63],
             [0.64], [0.65], [0.66], [0.67], [0.68], [0.69], [0.70], [0.71],
             [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78], [0.79],
             [0.80], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87],
             [0.88], [0.89], [0.90], [0.91], [0.92], [0.93], [0.94], [0.95],
             [0.96], [0.97], [0.98], [0.99], [1.00], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.10], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16], [1.17], [1.18], [1.19],
             [1.20], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27],
             [1.28], [1.29], [1.30], [1.31], [1.32], [1.33], [1.34], [1.35],
             [1.36], [1.37], [1.38], [1.39], [1.40], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.50],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57], [1.58], [1.59], [1.60], [1.61], [1.62], [1.63],
             [1.64], [1.65], [1.66], [1.67], [1.68],
             [1.69], [1.70], [1.71], [1.72], [1.73], [1.74], [1.75], [1.76],
             [1.77], [1.78], [1.79], [1.80], [1.81],
             [1.82], [1.83], [1.84], [1.85], [1.86], [1.87], [1.88], [1.89],
             [1.90], [1.91], [1.92], [1.93], [1.94],
             [1.95], [1.96], [1.97], [1.98], [1.99], [2.00], [2.01], [2.02],
             [2.03], [2.04], [2.05], [2.06], [2.07],
             [2.08], [2.09], [2.10], [2.11], [2.12], [2.13], [2.14], [2.15],
             [2.16], [2.17], [2.18], [2.19], [2.20],
             [2.21], [2.22], [2.23], [2.24], [2.25], [2.26], [2.27], [2.28],
             [2.29], [2.30], [2.31], [2.32], [2.33],
             [2.34], [2.35], [2.36], [2.37], [2.38], [2.39], [2.40], [2.41],
             [2.42], [2.43], [2.44], [2.45], [2.46],
             [2.47], [2.48], [2.49], [2.50], [2.51], [2.52], [2.53], [2.54],
             [2.55], [2.56], [2.57], [2.58], [2.59],
             [2.60], [2.61], [2.62], [2.63], [2.64], [2.65], [2.66], [2.67],
             [2.68], [2.69], [2.70], [2.71], [2.72],
             [2.73], [2.74], [2.75], [2.76], [2.77], [2.78], [2.79], [2.80],
             [2.81], [2.82], [2.83], [2.84], [2.85],
             [2.86], [2.87], [2.88], [2.89], [2.90], [2.91], [2.92], [2.93],
             [2.94], [2.95], [2.96], [2.97], [2.98],
             [2.99], [3.00], [3.01], [3.02], [3.03], [3.04], [3.05], [3.06],
             [3.07], [3.08], [3.09], [3.10], [3.11],
             [3.12], [3.13], [3.14], [3.15], [3.16], [3.17], [3.18], [3.19],
             [3.20], [3.21], [3.22], [3.23], [3.24],
             [3.25], [3.26], [3.27], [3.28], [3.29], [3.30], [3.31], [3.32],
             [3.33], [3.34], [3.35], [3.36], [3.37],
             [3.38], [3.39], [3.40], [3.41], [3.42], [3.43], [3.44], [3.45],
             [3.46], [3.47], [3.48], [3.49], [3.50],
             [3.51], [3.52], [3.53], [3.54], [3.55], [3.56], [3.57], [3.58],
             [3.59], [3.60], [3.61], [3.62], [3.63],
             [3.64], [3.65], [3.66], [3.67], [3.68], [3.69], [3.70], [3.71],
             [3.72], [3.73], [3.74], [3.75], [3.76],
             [3.77], [3.78], [3.79], [3.80], [3.81], [3.82], [3.83], [3.84],
             [3.85], [3.86], [3.87], [3.88], [3.89],
             [3.90], [3.91], [3.92], [3.93], [3.94], [3.95], [3.96], [3.97],
             [3.98], [3.99], [4.00], [4.01], [4.02],
             [4.03], [4.04], [4.05], [4.06], [4.07], [4.08], [4.09], [4.10],
             [4.11], [4.12], [4.13], [4.14], [4.15],
             [4.16], [4.17], [4.18], [4.19], [4.20], [4.21], [4.22], [4.23],
             [4.24], [4.25], [4.26], [4.27], [4.28],
             [4.29], [4.30], [4.31], [4.32], [4.33], [4.34], [4.35], [4.36],
             [4.37], [4.38], [4.39], [4.40], [4.41],
             [4.42], [4.43], [4.44], [4.45], [4.46], [4.47], [4.48], [4.49],
             [4.50], [4.51], [4.52], [4.53], [4.54],
             [4.55], [4.56], [4.57], [4.58], [4.59], [4.60], [4.61], [4.62],
             [4.63], [4.64], [4.65], [4.66], [4.67],
             [4.68], [4.69], [4.70], [4.71], [4.72], [4.73], [4.74], [4.75],
             [4.76], [4.77], [4.78], [4.79], [4.80],
             [4.81], [4.82], [4.83], [4.84], [4.85], [4.86], [4.87], [4.88],
             [4.89], [4.90], [4.91], [4.92], [4.93],
             [4.94], [4.95], [4.96], [4.97], [4.98], [4.99], [5.00], [5.01],
             [5.02], [5.03], [5.04], [5.05], [5.06],
             [5.07], [5.08], [5.09], [5.10], [5.11], [5.12], [5.13], [5.14],
             [5.15], [5.16], [5.17], [5.18], [5.19],
             [5.20], [5.21], [5.22], [5.23], [5.24], [5.25], [5.26], [5.27],
             [5.28], [5.29], [5.30], [5.31], [5.32],
             [5.33], [5.34], [5.35], [5.36], [5.37], [5.38], [5.39], [5.40],
             [5.41], [5.42], [5.43], [5.44], [5.45],
             [5.46], [5.47], [5.48], [5.49], [5.50], [5.51], [5.52], [5.53],
             [5.54], [5.55], [5.56], [5.57], [5.58],
             [5.59], [5.60], [5.61], [5.62], [5.63], [5.64], [5.65], [5.66],
             [5.67], [5.68], [5.69], [5.70], [5.71],
             [5.72], [5.73], [5.74], [5.75], [5.76], [5.77], [5.78], [5.79],
             [5.80], [5.81], [5.82], [5.83], [5.84],
             [5.85], [5.86], [5.87], [5.88], [5.89], [5.90], [5.91], [5.92],
             [5.93], [5.94], [5.95], [5.96], [5.97],
             [5.98], [5.99], [6.00], [6.01], [6.02], [6.03], [6.04], [6.05],
             [6.06], [6.07], [6.08], [6.09], [6.10],
             [6.11], [6.12], [6.13], [6.14], [6.15], [6.16], [6.17], [6.18],
             [6.19], [6.20], [6.21], [6.22], [6.23],
             [6.24], [6.25], [6.26], [6.27], [6.28]]
    sin_Y = [[0.0], [0.009999833334166664], [0.01999866669333308],
             [0.02999550020249566],
             [0.03998933418663416], [0.04997916927067833],
             [0.059964006479444595], [0.06994284733753277],
             [0.0799146939691727], [0.08987854919801104],
             [0.09983341664682815], [0.10977830083717481],
             [0.11971220728891936], [0.12963414261969486],
             [0.1395431146442365], [0.14943813247359922],
             [0.15931820661424598], [0.16918234906699603],
             [0.17902957342582418], [0.18885889497650057],
             [0.19866933079506122], [0.20845989984609956],
             [0.21822962308086932], [0.2279775235351884],
             [0.23770262642713458], [0.24740395925452294],
             [0.2570805518921551], [0.26673143668883115],
             [0.27635564856411376], [0.28595222510483553],
             [0.29552020666133955], [0.3050586364434435],
             [0.31456656061611776], [0.32404302839486837],
             [0.3334870921408144], [0.34289780745545134],
             [0.35227423327508994], [0.361615431964962], [0.3709204694129827],
             [0.3801884151231614],
             [0.3894183423086505], [0.3986093279844229], [0.40776045305957015],
             [0.41687080242921076],
             [0.4259394650659996], [0.43496553411123023], [0.4439481069655198],
             [0.4528862853790683],
             [0.4617791755414829], [0.470625888171158], [0.479425538604203],
             [0.48817724688290753],
             [0.49688013784373675], [0.5055333412048469], [0.5141359916531132],
             [0.5226872289306592],
             [0.5311861979208834], [0.5396320487339692], [0.5480239367918736],
             [0.5563610229127838],
             [0.5646424733950354], [0.5728674601004813], [0.5810351605373051],
             [0.5891447579422695],
             [0.5971954413623921], [0.6051864057360395], [0.6131168519734338],
             [0.6209859870365597],
             [0.6287930240184686], [0.6365371822219679], [0.644217687237691],
             [0.6518337710215366],
             [0.6593846719714731], [0.6668696350036979], [0.674287911628145],
             [0.6816387600233342],
             [0.6889214451105513], [0.6961352386273567], [0.7032794192004101],
             [0.7103532724176078],
             [0.7173560908995228], [0.7242871743701425], [0.7311458297268958],
             [0.7379313711099627],
             [0.7446431199708593], [0.7512804051402926], [0.757842562895277],
             [0.7643289370255051],
             [0.7707388788989693], [0.7770717475268238], [0.7833269096274833],
             [0.7895037396899504],
             [0.795601620036366], [0.8016199408837772], [0.8075581004051142],
             [0.8134155047893737],
             [0.8191915683009983], [0.82488571333845], [0.8304973704919705],
             [0.8360259786005204],
             [0.8414709848078965], [0.8468318446180152], [0.8521080219493629],
             [0.8572989891886034],
             [0.8624042272433384], [0.8674232255940169], [0.8723554823449863],
             [0.8772005042746817],
             [0.8819578068849475], [0.8866269144494873], [0.8912073600614354],
             [0.8956986856800477],
             [0.9001004421765051], [0.9044121893788258], [0.9086334961158832],
             [0.912763940260521],
             [0.9168031087717669], [0.9207505977361357], [0.9246060124080203],
             [0.9283689672491666],
             [0.9320390859672263], [0.9356160015533859], [0.9390993563190676],
             [0.9424888019316975],
             [0.945783999449539], [0.9489846193555862], [0.9520903415905158],
             [0.9551008555846923],
             [0.9580158602892249], [0.9608350642060727], [0.963558185417193],
             [0.9661849516127341],
             [0.9687151001182652], [0.9711483779210446], [0.9734845416953194],
             [0.9757233578266591],
             [0.9778646024353163], [0.9799080613986142], [0.9818535303723597],
             [0.9837008148112766],
             [0.9854497299884601], [0.9871001010138504], [0.9886517628517197],
             [0.9901045603371778],
             [0.9914583481916864], [0.9927129910375885], [0.9938683634116449],
             [0.9949243497775809],
             [0.99588084453764], [0.9967377520431434], [0.9974949866040544],
             [0.9981524724975481],
             [0.998710143975583], [0.999167945271476], [0.9995258306054791],
             [0.999783764189357],
             [0.9999417202299663], [0.9999996829318346], [0.9999576464987401],
             [0.9998156151342908],
             [0.9995736030415051], [0.9992316344213905], [0.998789743470524],
             [0.9982479743776325],
             [0.9976063813191737], [0.9968650284539189], [0.9960239899165367],
             [0.9950833498101802],
             [0.994043202198076], [0.9929036510941185], [0.9916648104524686],
             [0.990326804156158],
             [0.9888897660047015], [0.9873538397007164], [0.9857191788355535],
             [0.9839859468739369],
             [0.9821543171376185], [0.9802244727880455], [0.9781966068080447],
             [0.9760709219825242],
             [0.9738476308781951], [0.9715269558223153], [0.9691091288804563],
             [0.9665943918332975],
             [0.9639829961524481], [0.9612752029752999], [0.9584712830789142],
             [0.955571516852944],
             [0.9525761942715953], [0.9494856148646305], [0.9463000876874145],
             [0.9430199312900106],
             [0.9396454736853249], [0.9361770523163061], [0.9326150140222005],
             [0.9289597150038693],
             [0.9252115207881683], [0.9213708061913954], [0.9174379552818098],
             [0.9134133613412252],
             [0.9092974268256817], [0.905090563325201], [0.9007931915226273],
             [0.89640574115156],
             [0.8919286509533796], [0.8873623686333755], [0.8827073508159741],
             [0.8779640629990781],
             [0.8731329795075164], [0.8682145834456126], [0.8632093666488738],
             [0.8581178296348089],
             [0.8529404815528763], [0.8476778401335698], [0.8423304316366457],
             [0.8368987907984978],
             [0.8313834607786832], [0.8257849931056082], [0.8201039476213742],
             [0.814340892425796],
             [0.8084964038195901], [0.8025710662467473], [0.7965654722360865],
             [0.7904802223420048],
             [0.7843159250844198], [0.7780731968879213], [0.771752662020126],
             [0.7653549525292536],
             [0.7588807081809221], [0.7523305763941708], [0.7457052121767203],
             [0.7390052780594709],
             [0.7322314440302515], [0.7253843874668195], [0.7184647930691263],
             [0.7114733527908443],
             [0.7044107657701761], [0.6972777382599378], [0.6900749835569364],
             [0.6828032219306397],
             [0.675463180551151], [0.6680555934164909], [0.6605812012792007],
             [0.6530407515722648],
             [0.6454349983343707], [0.6377647021345036], [0.6300306299958922],
             [0.6222335553193046],
             [0.6143742578057118], [0.6064535233783147], [0.5984721441039564],
             [0.590430918113913],
             [0.5823306495240819], [0.5741721483545726], [0.5659562304487028],
             [0.557683717391417],
             [0.5493554364271266], [0.5409722203769886], [0.5325349075556212],
             [0.5240443416872761],
             [0.5155013718214642], [0.5069068522480534], [0.49826164241183857],
             [0.4895666068265995],
             [0.4808226149886482], [0.47203054128988264], [0.4631912649303451],
             [0.45430566983030646],
             [0.44537464454187115], [0.4363990821601263], [0.4273798802338298],
             [0.418317940675659],
             [0.4092141696720173], [0.4000694775924195], [0.3908847788984522],
             [0.38166099205233167],
             [0.3723990394250557], [0.3630998472041683], [0.3537643453011431],
             [0.34439346725839],
             [0.33498815015590505], [0.32554933451756], [0.3160779642170538],
             [0.30657498638352293],
             [0.2970413513068324], [0.2874780123425444], [0.2778859258165868],
             [0.2682660509296179],
             [0.2586193496611108], [0.24894678667315256], [0.2392493292139824],
             [0.2295279470212642],
             [0.21978361222511694], [0.21001729925089913],
             [0.2002299847217705], [0.19042264736102704],
             [0.18059626789423291], [0.17075182895114532],
             [0.16089031496745576], [0.15101271208634381],
             [0.1411200080598672], [0.13121319215018423],
             [0.12129325503062975], [0.11136118868665003],
             [0.10141798631660187], [0.0914646422324372],
             [0.08150215176026913], [0.0715315111408437],
             [0.06155371742991315], [0.05156976839853464],
             [0.04158066243329049], [0.031587398436453896],
             [0.02159097572609596], [0.011592393936158277],
             [0.0015926529164868282], [-0.008407247367148617],
             [-0.01840630693305381], [-0.02840352588360379],
             [-0.03839790450523538], [-0.04838844336841414],
             [-0.058374143427580086], [-0.06835400612104778],
             [-0.0783270334708653], [-0.0882922281826076],
             [-0.09824859374510868], [-0.10819513453010837],
             [-0.11813085589181736], [-0.12805476426637968],
             [-0.13796586727122687], [-0.14786317380431852],
             [-0.1577456941432482], [-0.16761244004421832],
             [-0.17746242484086017], [-0.1872946635429032],
             [-0.19710817293466987], [-0.20690197167339977],
             [-0.21667508038737965], [-0.22642652177388317],
             [-0.23615532069689701], [-0.24586050428463704],
             [-0.2555411020268312], [-0.2651961458717734],
             [-0.274824670323124], [-0.28442571253646254],
             [-0.2939983124155676], [-0.30354151270842933],
             [-0.31305435910297025], [-0.322535900322479],
             [-0.3319851882207341], [-0.34140127787682095],
             [-0.35078322768961984], [-0.36013009947196817],
             [-0.3694409585444771], [-0.3787148738289976],
             [-0.3879509179417303], [-0.3971481672859598],
             [-0.4063057021444168], [-0.4154226067712459],
             [-0.4244979694835826], [-0.43353088275271773],
             [-0.44252044329485246], [-0.45146575216142315],
             [-0.4603659148289983], [-0.46922004128872713],
             [-0.47802724613534286], [-0.4867866486556994],
             [-0.495497372916845], [-0.5041585478536115],
             [-0.5127693073557238], [-0.5213287903544065],
             [-0.5298361409084934], [-0.5382905082900177],
             [-0.5466910470692872], [-0.5550369171994238], [-0.56332728410037],
             [-0.5715613187423438],
             [-0.5797381977287428], [-0.5878571033784827],
             [-0.5959172238077639], [-0.6039177530112606],
             [-0.6118578909427189], [-0.6197368435949633],
             [-0.6275538230792934], [-0.6353080477042756],
             [-0.6429987420539088], [-0.6506251370651673],
             [-0.6581864701049049], [-0.6656819850461192],
             [-0.6731109323435617], [-0.680472569108694],
             [-0.6877661591839738], [-0.694990973216472],
             [-0.7021462887308054], [-0.709231390201386],
             [-0.7162455691239705], [-0.7231881240865121],
             [-0.7300583608392995], [-0.7368555923643834],
             [-0.7435791389442745], [-0.7502283282299189],
             [-0.7568024953079282], [-0.7633009827670734],
             [-0.7697231407640238], [-0.7760683270883323],
             [-0.7823359072266527], [-0.7885252544261949],
             [-0.7946357497573968], [-0.8006667821758177],
             [-0.8066177485832405], [-0.8124880538879842],
             [-0.8182771110644103], [-0.8239843412116258],
             [-0.8296091736113709], [-0.8351510457850935],
             [-0.8406094035501945], [-0.8459837010754465],
             [-0.8512734009355745], [-0.856477974165001],
             [-0.8615969003107404], [-0.8666296674844443],
             [-0.8715757724135881], [-0.8764347204918014],
             [-0.8812060258283253], [-0.8858892112966027],
             [-0.8904838085819885], [-0.8949893582285835],
             [-0.8994054096851777], [-0.9037315213503053],
             [-0.9079672606164054], [-0.9121122039130803],
             [-0.9161659367494549], [-0.9201280537556237],
             [-0.9239981587231879], [-0.9277758646448755],
             [-0.9314607937532425], [-0.9350525775584491],
             [-0.9385508568851079], [-0.941955281908201],
             [-0.9452655121880633], [-0.9484812167044256],
             [-0.9516020738895161], [-0.9546277716602164],
             [-0.9575580074492711], [-0.9603924882355434],
             [-0.9631309305733167], [-0.9657730606206388],
             [-0.9683186141667072], [-0.9707673366582883],
             [-0.9731189832251739], [-0.9753733187046665],
             [-0.977530117665097], [-0.9795891644283669],
             [-0.9815502530915153], [-0.9834131875473108],
             [-0.9851777815038595], [-0.9868438585032365],
             [-0.9884112519391305], [-0.9898798050735039],
             [-0.991249371052267], [-0.9925198129199632],
             [-0.9936910036334644], [-0.9947628260746756],
             [-0.9957351730622453], [-0.9966079473622855],
             [-0.9973810616980933], [-0.9980544387588794],
             [-0.9986280112074989], [-0.9991017216871848],
             [-0.999475522827284], [-0.999749377247994], [-0.9999232575641008],
             [-0.999997146387718],
             [-0.9999710363300245], [-0.9998449300020044],
             [-0.9996188400141854], [-0.999292788975378],
             [-0.9988668094904142], [-0.9983409441568876],
             [-0.9977152455608933], [-0.9969897762717695],
             [-0.9961646088358407], [-0.9952398257691627],
             [-0.9942155195492713], [-0.9930917926059354],
             [-0.9918687573109126], [-0.9905465359667133],
             [-0.9891252607943698], [-0.9876050739202153],
             [-0.9859861273616704], [-0.9842685830120416],
             [-0.9824526126243325], [-0.9805383977940689],
             [-0.9785261299411385], [-0.9764160102906497],
             [-0.9742082498528091], [-0.9719030694018208],
             [-0.9695006994538088], [-0.967001380243766],
             [-0.9644053617015305], [-0.9617129034267934],
             [-0.9589242746631385], [-0.9560397542711181],
             [-0.9530596307003677], [-0.9499842019607608],
             [-0.9468137755926089], [-0.9435486686359066],
             [-0.9401892075986287], [-0.9367357284240789],
             [-0.9331885764572976], [-0.9295481064105251],
             [-0.9258146823277325], [-0.9219886775482162],
             [-0.918070474669267], [-0.914060465507907], [-0.9099590510617106],
             [-0.9057666414687044],
             [-0.9014836559663548], [-0.8971105228496424],
             [-0.8926476794282346], [-0.8880955719827542],
             [-0.8834546557201531], [-0.8787253947281898],
             [-0.8739082619290224], [-0.8690037390319161],
             [-0.8640123164850744], [-0.8589344934265921],
             [-0.8537707776345433], [-0.8485216854762045],
             [-0.8431877418564168], [-0.8377694801650978],
             [-0.8322674422239013], [-0.8266821782320363],
             [-0.8210142467112471], [-0.8152642144499636],
             [-0.8094326564466194], [-0.8035201558521559],
             [-0.7975273039117043], [-0.791454699905466],
             [-0.7853029510887806], [-0.7790726726314032],
             [-0.7727644875559871], [-0.7663790266757844],
             [-0.759916928531561], [-0.7533788393277466],
             [-0.7467654128678123], [-0.7400773104888945],
             [-0.7333152009956565], [-0.726479760593413],
             [-0.7195716728205075], [-0.7125916284799616],
             [-0.7055403255703919], [-0.6984184692162135],
             [-0.6912267715971271], [-0.6839659518769006],
             [-0.6766367361314568], [-0.669239857276262],
             [-0.6617760549930376], [-0.6542460756557913],
             [-0.6466506722561834], [-0.6389906043282237],
             [-0.6312666378723216], [-0.6234795452786853],
             [-0.6156301052500863], [-0.6077191027239858],
             [-0.5997473287940438], [-0.5917155806310094],
             [-0.5836246614030073], [-0.575475380195217],
             [-0.5672685519289686], [-0.5590049972802488],
             [-0.5506855425976376], [-0.5423110198196697],
             [-0.5338822663916443], [-0.5254001251818792],
             [-0.5168654443974288], [-0.5082790774992583],
             [-0.49964188311690244], [-0.49095472496260173],
             [-0.48221847174493154], [-0.473433997081935],
             [-0.46460217941375737], [-0.4557239019148055],
             [-0.44680005240542997], [-0.4378315232631469],
             [-0.42881921133339584], [-0.41976401783985967],
             [-0.41066684829434086], [-0.4015286124062146],
             [-0.39235022399145386], [-0.38313260088125134],
             [-0.373876664830236], [-0.3645833414243013],
             [-0.35525355998804264], [-0.34588825349182883],
             [-0.3364883584585042], [-0.32705481486974064],
             [-0.31758856607203484], [-0.3080905586823781],
             [-0.2985617424935936], [-0.2890030703793611],
             [-0.27941549819892586], [-0.26979998470151617],
             [-0.2601574914304689], [-0.25048898262707486],
             [-0.2407954251341592], [-0.23107778829939224],
             [-0.22133704387835954], [-0.21157416593738504],
             [-0.2017901307561289], [-0.19198591672995502],
             [-0.1821625042720959], [-0.17232087571561025],
             [-0.1624620152151542], [-0.15258690864856114],
             [-0.1426965435182586], [-0.13279190885251674],
             [-0.12287399510655005], [-0.11294379406346737],
             [-0.10300229873509785], [-0.0930505032626889],
             [-0.08308940281749641], [-0.0731199935012631],
             [-0.06314327224661277], [-0.053160236717356125],
             [-0.04317188520872868], [-0.03317921654755682],
             [-0.023183229992379453], [-0.01318492513352214],
             [-0.0031853017931379904]]

    x, y = preprocess_data(sin_X, sin_Y)

    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)

    data = NNData(x, y, 0.7)
    network.train(data, 20001, order=NNData.Order.RANDOM)
    x, y = network.test(data)

    plt.plot(sin_X, sin_Y, "r", label="Actual")
    plt.plot(x, y, "y", label="Predicted, tf = 0.7")
    plt.title("Sin curve from 0 to 6.28 radians, actual vs. predicted values")
    plt.xlabel("Radian")
    plt.legend()
    plt.show()


def preprocess_data(x, y):
    for _ in range(len(x)):
        x[_][0] = x[_][0] + 1
        y[_][0] = y[_][0] + 1
    return x, y


def explore_iris_epoch():
    test_epochs = [2001, 5001, 10001, 20001]
    # create a list of arrays corresponding to each element in test_epochs
    sums_y = []
    # create a list of x val arrays corresponding to each element in epoch
    x_val = []

    for i in range(len(test_epochs)):
        x, y = run_iris(0.7, test_epochs[i])
        x_val.append(np.array(x))
        sums_y.append(np.array(y))

    avg_y = []
    for i in range(len(test_epochs)):
        for j in range(2):
            x, y = run_iris(0.7, test_epochs[i])
            sums_y[i] += np.array(y)

        avg_y_arr = sums_y[i] / 3
        avg_y.append(avg_y_arr)

    colors = ["or", "ob", "og", "oy"]

    for i in range(len(test_epochs)):
        plt.plot(x_val[i], avg_y[i], colors[i], label=str(test_epochs[i]))

    plt.title("Iris Dataset by Number of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def explore_iris_num_node():
    num_nodes = [1, 3, 5, 10]
    # create a list of arrays corresponding to each element in test_epochs
    sums_y = []
    # create a list of x val arrays corresponding to each element in epoch
    x_val = []

    for i in range(len(num_nodes)):
        x, y = run_iris(0.7, num_nodes[i])
        x_val.append(np.array(x))
        sums_y.append(np.array(y))

    avg_y = []
    for i in range(len(num_nodes)):
        for j in range(2):
            x, y = run_iris(0.7, num_nodes[i])
            sums_y[i] += np.array(y)

        avg_y_arr = sums_y[i] / 3
        avg_y.append(avg_y_arr)

    colors = ["r", "b", "g", "y"]

    for i in range(len(num_nodes)):
        plt.plot(x_val[i], avg_y[i], colors[i], label=str(num_nodes[i]))

    plt.title("Iris Dataset by Number of Neurodes in Hidden layer")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def run_iris_times_ten():
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k', '-c', '-m', '-y', '-k']
    for i in range(10):
        x, y = run_iris()
        plt.plot(x, y, colors[i])

    plt.title("Iris data set, RMSE over 10 repetitions")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.show()


def main():
    # extra_credit_1()
    # extra_credit_2()
    extra_credit_3()
    # explore_iris_epoch()
    # explore_iris_num_node()
    # run_iris_times_ten()
    # sin_output()
    # run_iris()
    # run_sin()
    # run_xor()
    # run_xor_bias()


if __name__ == "__main__":
    main()

"""
/Users/anne/PycharmProjects/IntSoftwrDsgn_Python/venv/bin/python 
/Users/anne/PycharmProjects/IntSoftwrDsgn_Python/assignment5.py
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.75                
3.50                0.00                0.79                
1.60                0.00                0.68                
0.60                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.75                
3.70                0.00                0.79                
1.50                0.00                0.67                
0.40                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.75                
2.30                1.00                0.79                
3.30                0.00                0.67                
1.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.75                
3.30                0.00                0.79                
6.00                1.00                0.66                
2.50                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.74                
3.80                0.00                0.79                
1.50                0.00                0.67                
0.30                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.74                
3.10                0.00                0.78                
5.10                1.00                0.66                
2.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.74                
2.50                1.00                0.78                
4.00                0.00                0.66                
1.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.74                
3.70                0.00                0.78                
1.50                0.00                0.66                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.74                
3.60                0.00                0.78                
6.10                1.00                0.65                
2.50                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.73                
3.00                1.00                0.77                
4.20                0.00                0.66                
1.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.73                
3.00                0.00                0.77                
5.80                1.00                0.65                
2.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.73                
2.80                1.00                0.77                
4.10                0.00                0.65                
1.30                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.72                
4.20                0.00                0.77                
1.40                0.00                0.65                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.72                
2.80                1.00                0.77                
4.00                0.00                0.64                
1.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.72                
2.60                1.00                0.77                
4.40                0.00                0.64                
1.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.71                
3.40                0.00                0.77                
1.50                0.00                0.63                
0.40                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.72                
3.30                0.00                0.77                
5.70                1.00                0.63                
2.10                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.71                
3.00                0.00                0.76                
5.20                1.00                0.63                
2.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.71                
3.00                0.00                0.76                
1.40                0.00                0.63                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.71                
2.80                0.00                0.76                
5.60                1.00                0.63                
2.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.70                
2.20                1.00                0.75                
4.50                0.00                0.63                
1.50                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.70                
3.20                0.00                0.75                
5.70                1.00                0.63                
2.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.69                
2.80                1.00                0.75                
4.50                0.00                0.63                
1.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.69                
3.00                0.00                0.75                
5.80                1.00                0.63                
1.60                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.69                
3.00                0.00                0.75                
6.10                1.00                0.63                
2.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.68                
3.40                0.00                0.74                
1.50                0.00                0.63                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.68                
2.40                1.00                0.74                
3.80                0.00                0.63                
1.10                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.68                
4.00                0.00                0.74                
1.20                0.00                0.62                
0.20                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.68                
3.00                1.00                0.74                
4.40                0.00                0.62                
1.40                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.67                
3.80                0.00                0.74                
1.70                0.00                0.61                
0.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.68                
2.80                0.00                0.73                
5.10                1.00                0.61                
2.40                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.67                
2.70                1.00                0.73                
3.90                0.00                0.61                
1.40                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.67                
3.00                0.00                0.73                
5.50                1.00                0.60                
1.80                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.66                
3.80                0.00                0.73                
6.70                1.00                0.61                
2.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.66                
2.90                1.00                0.72                
4.70                0.00                0.61                
1.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.65                
3.50                0.00                0.72                
1.40                0.00                0.60                
0.20                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.65                
3.10                0.00                0.72                
1.50                0.00                0.60                
0.10                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.66                
3.20                0.00                0.71                
1.30                0.00                0.59                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.66                
2.70                0.00                0.71                
4.90                1.00                0.59                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.66                
3.80                0.00                0.71                
1.60                0.00                0.59                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.66                
2.90                1.00                0.70                
4.30                0.00                0.59                
1.30                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.65                
3.00                0.00                0.70                
4.90                1.00                0.58                
1.80                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.65                
3.00                0.00                0.70                
5.20                1.00                0.59                
2.00                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.64                
2.70                0.00                0.69                
5.30                1.00                0.59                
1.90                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.64                
3.90                0.00                0.69                
1.70                0.00                0.59                
0.40                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.64                
2.80                0.00                0.68                
4.90                1.00                0.59                
2.00                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.64                
3.20                0.00                0.68                
1.20                0.00                0.59                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.64                
3.00                1.00                0.67                
4.50                0.00                0.59                
1.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.63                
3.20                1.00                0.68                
4.50                0.00                0.58                
1.50                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.63                
2.70                1.00                0.68                
5.10                0.00                0.58                
1.60                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.62                
3.00                1.00                0.68                
4.60                0.00                0.57                
1.40                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.62                
3.00                0.00                0.68                
5.10                1.00                0.57                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.61                
4.40                0.00                0.68                
1.50                0.00                0.57                
0.40                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.62                
2.50                0.00                0.67                
4.50                1.00                0.57                
1.70                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.61                
2.80                0.00                0.67                
5.10                1.00                0.57                
1.50                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.61                
3.20                1.00                0.66                
4.80                0.00                0.57                
1.80                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.60                
2.70                1.00                0.67                
4.10                0.00                0.57                
1.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.60                
2.90                0.00                0.67                
5.60                1.00                0.56                
1.80                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.59                
3.10                0.00                0.66                
5.40                1.00                0.57                
2.10                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.59                
2.60                1.00                0.66                
4.00                0.00                0.57                
1.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.58                
3.00                0.00                0.66                
4.80                1.00                0.57                
1.80                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.57                
3.20                1.00                0.66                
4.70                0.00                0.57                
1.40                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.57                
2.90                0.00                0.66                
1.40                0.00                0.56                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.57                
3.40                0.00                0.65                
1.60                0.00                0.56                
0.40                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.58                
2.90                1.00                0.65                
4.60                0.00                0.55                
1.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.57                
3.60                0.00                0.65                
1.40                0.00                0.55                
0.20                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.58                
3.00                0.00                0.65                
5.90                1.00                0.54                
2.10                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.57                
2.50                1.00                0.64                
3.00                0.00                0.55                
1.10                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.57                
3.70                0.00                0.64                
1.50                0.00                0.54                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.57                
2.20                0.00                0.64                
5.00                1.00                0.54                
1.50                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.56                
2.50                0.00                0.63                
5.80                1.00                0.54                
1.80                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.56                
3.10                0.00                0.63                
1.50                0.00                0.55                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.56                
2.80                0.00                0.62                
6.70                1.00                0.54                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.56                
2.50                0.00                0.62                
5.00                1.00                0.55                
1.90                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.55                
3.20                0.00                0.61                
1.40                0.00                0.55                
0.20                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.56                
3.50                0.00                0.61                
1.50                0.00                0.55                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.56                
3.20                0.00                0.60                
5.10                1.00                0.54                
2.00                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.56                
3.40                1.00                0.60                
4.50                0.00                0.54                
1.60                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.55                
2.20                1.00                0.60                
4.00                0.00                0.54                
1.00                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.55                
2.30                0.00                0.60                
1.30                0.00                0.53                
0.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.55                
3.10                0.00                0.60                
5.50                1.00                0.53                
1.80                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.54                
3.40                0.00                0.60                
1.40                0.00                0.53                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.55                
2.80                1.00                0.59                
4.70                0.00                0.53                
1.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.54                
2.50                1.00                0.59                
3.90                0.00                0.52                
1.10                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.54                
3.40                0.00                0.60                
5.40                1.00                0.52                
2.30                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.53                
3.00                0.00                0.59                
6.60                1.00                0.52                
2.10                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.53                
3.40                0.00                0.59                
1.60                0.00                0.53                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.53                
2.70                0.00                0.58                
5.10                1.00                0.52                
1.90                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.53                
3.80                0.00                0.58                
6.40                1.00                0.53                
2.00                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.52                
3.20                0.00                0.57                
6.00                1.00                0.53                
1.80                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.52                
3.10                0.00                0.57                
1.50                0.00                0.54                
0.10                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.52                
3.50                0.00                0.56                
1.40                0.00                0.53                
0.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.53                
2.60                0.00                0.56                
6.90                1.00                0.53                
2.30                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.52                
3.00                0.00                0.55                
1.40                0.00                0.53                
0.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.53                
3.40                0.00                0.55                
5.60                1.00                0.53                
2.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.52                
3.20                0.00                0.54                
5.30                1.00                0.53                
2.30                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.52                
3.00                0.00                0.54                
5.50                1.00                0.53                
2.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.51                
2.30                1.00                0.53                
4.40                0.00                0.54                
1.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.51                
2.50                0.00                0.54                
5.00                1.00                0.53                
2.00                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.50                
2.80                0.00                0.53                
4.80                1.00                0.54                
1.80                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.50                
3.60                0.00                0.53                
1.00                0.00                0.54                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.50                
3.30                0.00                0.52                
1.40                0.00                0.54                
0.20                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.51                
2.90                0.00                0.52                
6.30                1.00                0.53                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.50                
2.90                1.00                0.51                
4.20                0.00                0.54                
1.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.50                
2.80                0.00                0.52                
5.60                1.00                0.53                
2.10                
Epoch 0 RMSE =  0.5624290195160562
Epoch 100 RMSE =  0.3299115357496599
Epoch 200 RMSE =  0.30572646051551117
Epoch 300 RMSE =  0.2974779785543313
Epoch 400 RMSE =  0.2892674074623774
Epoch 500 RMSE =  0.2888549210702531
Epoch 600 RMSE =  0.2897611411664567
Epoch 700 RMSE =  0.28460727120407725
Epoch 800 RMSE =  0.2853494415682718
Epoch 900 RMSE =  0.2864526709570134
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.28                
6.60                1.00                0.91                
2.10                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.91                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.27                
5.90                1.00                0.90                
2.10                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.91                
4.00                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.27                
6.00                1.00                0.80                
1.80                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.27                
5.70                1.00                0.89                
2.10                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.27                
4.80                1.00                0.83                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.27                
2.50                1.00                0.27                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.02                
2.50                1.00                0.27                
3.90                0.00                0.08                
1.10                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.91                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.01                
2.60                1.00                0.27                
4.00                0.00                0.10                
1.20                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.28                
5.10                1.00                0.87                
1.80                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.27                
4.50                0.00                0.50                
1.50                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.51                
1.60                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.91                
3.90                0.00                0.28                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.02                
2.90                1.00                0.28                
4.30                0.00                0.06                
1.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.28                
5.10                1.00                0.91                
1.90                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.91                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.27                
5.50                1.00                0.88                
1.80                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.91                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.27                
4.60                0.00                0.30                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.65                
1.50                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.09                
2.70                1.00                0.28                
4.10                0.00                0.01                
1.00                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.91                
3.80                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.91                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.91                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.27                
5.60                1.00                0.92                
2.40                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.91                
3.80                0.00                0.28                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.27                
4.80                0.00                0.81                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.91                
3.20                0.00                0.28                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.03                
3.20                1.00                0.28                
4.50                0.00                0.05                
1.50                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.83                
1.60                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.28                
4.50                1.00                0.87                
1.70                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.28                
5.00                1.00                0.91                
2.00                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.09                
2.90                1.00                0.28                
4.60                0.00                0.02                
1.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.01                
2.50                1.00                0.28                
4.00                0.00                0.16                
1.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.91                
3.80                0.00                0.29                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.91                
3.60                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.08                
2.40                1.00                0.28                
3.80                0.00                0.02                
1.10                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.29                
6.10                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.91                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.91                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.28                
4.90                1.00                0.71                
1.80                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.28                
6.70                1.00                0.88                
2.20                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.91                
4.40                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.13                
3.00                1.00                0.28                
4.20                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.08                
2.30                1.00                0.28                
3.30                0.00                0.02                
1.00                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.04                
2.80                1.00                0.29                
4.00                0.00                0.03                
1.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.29                
5.10                1.00                0.92                
2.40                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.29                
5.70                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.29                
6.00                1.00                0.92                
2.50                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.28                
4.90                1.00                0.91                
2.00                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.28                
3.90                0.00                0.31                
1.40                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.29                
5.20                1.00                0.85                
2.00                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.05                
2.20                1.00                0.28                
4.00                0.00                0.02                
1.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.29                
6.90                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.29                
4.50                0.00                0.77                
1.50                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.29                
6.30                1.00                0.83                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.91                
4.20                0.00                0.29                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.29                
5.10                1.00                0.67                
2.00                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.91                
3.50                0.00                0.29                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.90                
2.30                0.00                0.28                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.91                
3.50                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.82                
1.80                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.91                
3.20                0.00                0.28                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.01                
2.80                1.00                0.28                
4.50                0.00                0.13                
1.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.91                
3.60                0.00                0.28                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.28                
5.00                1.00                0.89                
1.90                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.28                
5.40                1.00                0.89                
2.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.28                
6.70                1.00                0.91                
2.00                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.91                
2.90                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.03                
2.80                1.00                0.28                
4.70                0.00                0.04                
1.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.91                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.28                
6.40                1.00                0.62                
2.00                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.91                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.27                
5.00                1.00                0.90                
1.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.27                
5.30                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.01                
3.00                1.00                0.27                
4.40                0.00                0.09                
1.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.27                
4.70                0.00                0.43                
1.40                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.91                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.28                
4.90                1.00                0.81                
1.80                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.91                
3.40                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.91                
3.30                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.27                
5.10                1.00                0.88                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.27                
4.80                1.00                0.72                
1.80                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.27                
5.40                1.00                0.89                
2.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.27                
4.40                0.00                0.46                
1.30                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.48                
3.20                1.00                0.27                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.92                
2.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.91                
3.50                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.91                
3.20                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.27                
6.10                1.00                0.92                
2.50                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.92                
2.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.27                
5.30                1.00                0.90                
1.90                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.27                
5.50                1.00                0.90                
2.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.91                
3.50                0.00                0.27                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.02                
2.80                1.00                0.27                
4.10                0.00                0.08                
1.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.02                
2.90                1.00                0.27                
4.20                0.00                0.06                
1.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.91                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.01                
3.40                1.00                0.27                
4.50                0.00                0.08                
1.60                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.27                
5.60                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.27                
4.40                0.00                0.23                
1.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.90                
1.80                
Epoch 1000 RMSE =  0.2867040522020686
Epoch 1100 RMSE =  0.27906468785190375
Epoch 1200 RMSE =  0.28433555309352493
Epoch 1300 RMSE =  0.28350002541568414
Epoch 1400 RMSE =  0.280593858384196
Epoch 1500 RMSE =  0.2830504701568988
Epoch 1600 RMSE =  0.28411599068931664
Epoch 1700 RMSE =  0.27845937951483646
Epoch 1800 RMSE =  0.28661358608388887
Epoch 1900 RMSE =  0.2837305186095547
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.27                
4.40                0.00                0.28                
1.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.93                
3.90                0.00                0.28                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.27                
4.90                1.00                0.60                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.80                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.93                
4.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.80                0.00                0.27                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.27                
4.50                0.00                0.04                
1.60                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.27                
5.70                1.00                0.91                
2.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.27                
4.50                0.00                0.14                
1.30                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.28                
6.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.20                1.00                0.88                
2.00                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.07                
2.30                1.00                0.27                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.04                
2.90                1.00                0.28                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.93                
3.80                0.00                0.28                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.02                
2.50                1.00                0.28                
3.90                0.00                0.01                
1.10                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.93                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.28                
6.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.28                
4.80                1.00                0.78                
1.80                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.28                
4.80                0.00                0.74                
1.80                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.28                
4.50                0.00                0.07                
1.50                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.93                
2.50                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.52                
2.50                1.00                0.28                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.01                
2.20                1.00                0.28                
4.00                0.00                0.02                
1.00                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.93                
4.40                0.00                0.29                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.03                
2.80                1.00                0.29                
4.00                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.29                
5.20                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.29                
5.60                1.00                0.93                
2.40                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.29                
5.50                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.93                
3.40                0.00                0.29                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.92                
2.30                0.00                0.28                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.93                
2.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.28                
6.70                1.00                0.91                
2.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.28                
5.10                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.50                0.00                0.28                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.28                
6.70                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.01                
3.20                1.00                0.28                
4.50                0.00                0.02                
1.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.86                
2.00                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.93                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.28                
5.00                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.27                
5.00                1.00                0.90                
1.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.27                
5.30                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.92                
2.90                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.27                
5.80                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.01                
2.90                1.00                0.27                
4.60                0.00                0.03                
1.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.93                
3.60                0.00                0.27                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.27                
4.00                0.00                0.23                
1.30                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.93                
3.40                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.27                
4.70                0.00                0.11                
1.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.93                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.93                
3.20                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.27                
5.40                1.00                0.88                
2.10                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.01                
2.80                1.00                0.27                
4.70                0.00                0.02                
1.20                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.93                
3.20                0.00                0.28                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.02                
2.40                1.00                0.27                
3.80                0.00                0.01                
1.10                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.28                
5.70                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.93                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.02                
2.60                1.00                0.28                
4.00                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.28                
4.50                1.00                0.89                
1.70                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.05                
3.00                1.00                0.28                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.28                
4.40                0.00                0.10                
1.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.93                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.28                
5.90                1.00                0.91                
2.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.54                
1.50                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.50                0.00                0.28                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.28                
6.40                1.00                0.88                
2.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.28                
6.10                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.27                
5.10                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.27                
4.90                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.01                
3.00                1.00                0.27                
4.20                0.00                0.02                
1.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.27                
4.60                0.00                0.34                
1.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.28                
5.30                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.93                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.28                
4.80                1.00                0.84                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.27                
5.00                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.30                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.93                
2.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.27                
5.50                1.00                0.87                
1.80                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.50                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.03                
3.20                1.00                0.27                
4.70                0.00                0.01                
1.40                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.60                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.27                
4.90                1.00                0.92                
2.00                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.27                
3.90                0.00                0.23                
1.40                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.68                
1.60                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.27                
5.60                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.05                
1.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.93                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.27                
6.10                1.00                0.93                
2.50                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.27                
5.10                0.00                0.88                
1.60                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.01                
2.80                1.00                0.27                
4.10                0.00                0.03                
1.30                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.93                
3.50                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.27                
6.30                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.93                
4.00                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.27                
4.50                0.00                0.81                
1.50                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.10                1.00                0.89                
2.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.93                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.29                
2.70                1.00                0.27                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.28                
6.00                1.00                0.86                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.27                
5.40                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.20                0.00                0.27                
1.20                0.00                0.00                
0.20                
Epoch 2000 RMSE =  0.28089267567545284
Epoch 2100 RMSE =  0.27970334043412926
Epoch 2200 RMSE =  0.28149069718943964
Epoch 2300 RMSE =  0.2829002245546702
Epoch 2400 RMSE =  0.28204964864225046
Epoch 2500 RMSE =  0.2804144606682507
Epoch 2600 RMSE =  0.2818355133520665
Epoch 2700 RMSE =  0.2824821253210589
Epoch 2800 RMSE =  0.278621978556403
Epoch 2900 RMSE =  0.281465462195543
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.05                
3.20                1.00                0.27                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.28                
4.40                0.00                0.06                
1.30                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.40                1.00                0.85                
2.10                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.28                
4.80                1.00                0.72                
1.80                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.28                
4.70                0.00                0.08                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.11                
3.20                1.00                0.28                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.28                
5.30                1.00                0.91                
1.90                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.93                
3.80                0.00                0.28                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.05                
2.30                1.00                0.28                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.28                
5.00                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.93                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.93                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.03                
2.80                1.00                0.28                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.93                
3.20                0.00                0.28                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.93                
3.20                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.11                
2.70                1.00                0.28                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.03                
3.00                1.00                0.28                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.29                
5.10                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.28                
6.10                1.00                0.93                
2.50                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.28                
4.40                0.00                0.16                
1.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.29                
4.90                1.00                0.78                
1.80                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.93                
3.40                0.00                0.29                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.28                
5.70                1.00                0.91                
2.10                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.93                
2.30                0.00                0.28                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.80                0.00                0.28                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.28                
4.50                1.00                0.91                
1.70                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.93                
3.50                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.80                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.28                
4.60                0.00                0.06                
1.40                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.93                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.86                
1.60                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.28                
6.10                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.01                
3.40                1.00                0.28                
4.50                0.00                0.01                
1.60                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.50                0.00                0.28                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.04                
3.00                1.00                0.28                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.93                
3.10                0.00                0.29                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.29                
5.70                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.60                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.28                
4.80                1.00                0.72                
1.80                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.93                
2.90                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.93                
3.40                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.28                
6.40                1.00                0.79                
2.00                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.01                
2.90                1.00                0.28                
4.30                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.28                
4.50                0.00                0.44                
1.50                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.28                
4.50                0.00                0.59                
1.50                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.29                
6.00                1.00                0.11                
1.80                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.29                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.93                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.28                
4.90                1.00                0.83                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.28                
3.90                0.00                0.07                
1.40                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.28                
4.00                0.00                0.05                
1.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.29                
5.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.29                
6.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.87                
2.00                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.28                
4.50                0.00                0.05                
1.30                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.29                
4.80                0.00                0.59                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.20                
2.80                1.00                0.29                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.29                
5.00                1.00                0.91                
2.00                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.93                
4.00                0.00                0.29                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.29                
5.60                1.00                0.92                
2.20                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.29                
5.10                1.00                0.66                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.93                
4.20                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.29                
6.70                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.28                
5.40                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.28                
6.70                1.00                0.88                
2.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.04                
2.50                1.00                0.28                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.28                
5.50                1.00                0.82                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.93                
4.40                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.28                
6.30                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.28                
6.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.02                
2.20                1.00                0.28                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.28                
4.90                1.00                0.92                
2.00                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.28                
5.10                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.81                
2.00                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.01                
2.60                1.00                0.27                
4.00                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.28                
5.60                1.00                0.93                
2.40                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.26                
2.50                1.00                0.28                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.28                
5.30                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.93                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.93                
3.90                0.00                0.28                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.30                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.93                
3.20                0.00                0.28                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.50                0.00                0.28                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.27                
5.60                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.27                
5.00                1.00                0.91                
1.50                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.93                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.93                
3.60                0.00                0.27                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.93                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.83                
1.60                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.26                
5.90                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.26                
3.80                0.00                0.03                
1.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.27                
5.10                1.00                0.80                
1.50                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.27                
4.70                0.00                0.18                
1.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.93                
3.40                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
2.90                1.00                0.27                
4.60                0.00                0.02                
1.30                
Epoch 3000 RMSE =  0.2796979199607335
Epoch 3100 RMSE =  0.28414990042644667
Epoch 3200 RMSE =  0.28126922824813244
Epoch 3300 RMSE =  0.28141929872266813
Epoch 3400 RMSE =  0.27937953210244615
Epoch 3500 RMSE =  0.2842186141646649
Epoch 3600 RMSE =  0.28261678889238545
Epoch 3700 RMSE =  0.2813160538464566
Epoch 3800 RMSE =  0.2804543391882028
Epoch 3900 RMSE =  0.2805485725191922
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.90                1.00                0.27                
4.30                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.28                
4.50                0.00                0.87                
1.50                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.28                
5.50                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.28                
5.10                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.28                
5.90                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.80                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.12                
2.50                1.00                0.27                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.50                0.00                0.28                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.28                
4.50                0.00                0.25                
1.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.93                
2.30                0.00                0.28                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.84                
2.00                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                1.00                0.28                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.06                
2.70                1.00                0.28                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.10                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.94                
3.90                0.00                0.28                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.91                
2.00                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.94                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.27                
5.10                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.27                
4.70                0.00                0.04                
1.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.87                
1.60                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.94                
3.50                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.94                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.94                
4.00                0.00                0.28                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.28                
5.60                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.94                
3.80                0.00                0.27                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.27                
4.90                1.00                0.85                
1.80                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.27                
4.80                0.00                0.67                
1.80                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.94                
2.90                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.08                
2.80                1.00                0.27                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.28                
6.10                1.00                0.90                
2.50                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.94                
3.60                0.00                0.27                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.27                
5.10                1.00                0.12                
1.50                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.01                
2.80                1.00                0.27                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.94                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
2.90                1.00                0.27                
4.60                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.28                
4.90                1.00                0.84                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.28                
5.30                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.94                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.27                
4.00                0.00                0.18                
1.30                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.94                
3.20                0.00                0.28                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.28                
5.00                1.00                0.90                
1.50                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.94                
3.40                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.27                
5.70                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.94                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.27                
5.00                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.27                
5.30                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.27                
5.00                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.27                
4.50                1.00                0.92                
1.70                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.27                
6.70                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.26                
6.10                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.26                
4.40                0.00                0.15                
1.20                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.94                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.27                
6.40                1.00                0.72                
2.00                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.26                
5.50                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.26                
6.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.26                
5.40                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.26                
4.50                0.00                0.43                
1.50                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.26                
5.80                1.00                0.45                
1.60                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.94                
3.20                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.20                0.00                0.26                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.50                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.26                
4.20                0.00                0.11                
1.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.26                
6.70                1.00                0.93                
2.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.26                
6.00                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.26                
4.40                0.00                0.62                
1.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.05                
3.00                1.00                0.26                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.27                
5.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.27                
5.40                1.00                0.89                
2.10                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.01                
2.20                1.00                0.26                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.01                
2.50                1.00                0.27                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.27                
4.60                0.00                0.02                
1.40                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.94                
4.40                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.27                
3.90                0.00                0.05                
1.40                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.94                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.27                
4.90                1.00                0.92                
2.00                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.80                0.00                0.27                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.27                
6.30                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.30                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.27                
5.60                1.00                0.93                
2.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.27                
4.70                0.00                0.06                
1.40                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.03                
3.00                1.00                0.27                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.27                
4.50                0.00                0.01                
1.60                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.08                
3.20                1.00                0.28                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.01                
2.40                1.00                0.28                
3.80                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.29                
6.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.94                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.60                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.94                
4.20                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.01                
2.60                1.00                0.28                
4.00                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.94                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.04                
2.30                1.00                0.28                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.28                
4.80                1.00                0.81                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.28                
4.80                1.00                0.79                
1.80                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.28                
5.70                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.40                
Epoch 4000 RMSE =  0.2841885073448731
Epoch 4100 RMSE =  0.2796925381400868
Epoch 4200 RMSE =  0.2781725706240991
Epoch 4300 RMSE =  0.2776945544364128
Epoch 4400 RMSE =  0.2801057663456193
Epoch 4500 RMSE =  0.27833336198700437
Epoch 4600 RMSE =  0.2804595205547374
Epoch 4700 RMSE =  0.2812956978117236
Epoch 4800 RMSE =  0.2786271565968754
Epoch 4900 RMSE =  0.279223410821305
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.28                
4.00                0.00                0.15                
1.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.94                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.20                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.94                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.30                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.27                
5.90                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.27                
4.40                0.00                0.14                
1.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.28                
5.00                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.27                
6.10                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.27                
4.70                0.00                0.08                
1.40                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.28                
4.90                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.01                
2.50                1.00                0.27                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.28                
5.50                1.00                0.87                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                1.00                0.28                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.50                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.28                
6.40                1.00                0.75                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.28                
5.60                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.89                
1.60                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.94                
4.40                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.28                
6.70                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.86                
2.00                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.28                
5.10                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.10                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.27                
3.90                0.00                0.12                
1.40                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.50                0.00                0.27                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.94                
3.80                0.00                0.27                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.27                
4.10                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.01                
2.90                1.00                0.28                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.28                
5.10                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.28                
4.90                1.00                0.88                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.94                
4.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.79                
1.60                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.94                
3.60                0.00                0.27                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.27                
4.50                1.00                0.93                
1.70                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.27                
5.70                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.60                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.27                
5.00                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.94                
4.00                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.27                
4.50                0.00                0.42                
1.50                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.94                
3.50                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.27                
5.10                1.00                0.34                
1.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.27                
5.60                1.00                0.95                
2.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.94                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
3.00                1.00                0.27                
4.40                0.00                0.04                
1.40                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.00                
3.20                1.00                0.27                
4.70                0.00                0.02                
1.40                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.94                
3.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.27                
5.10                1.00                0.95                
2.40                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.94                
3.40                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.27                
3.80                0.00                0.06                
1.10                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
2.90                1.00                0.27                
4.60                0.00                0.08                
1.30                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.27                
4.50                0.00                0.20                
1.60                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.60                1.00                0.28                
4.00                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.28                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.29                
4.90                1.00                0.88                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.00                
2.30                1.00                0.28                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.29                
5.40                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.29                
6.00                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.29                
4.50                0.00                0.87                
1.50                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.29                
4.70                0.00                0.04                
1.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.29                
6.70                1.00                0.93                
2.20                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.94                
3.40                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.94                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.94                
3.20                0.00                0.28                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.15                
2.50                1.00                0.28                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.29                
4.40                0.00                0.40                
1.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.94                
3.90                0.00                0.29                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.29                
4.60                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.30                
5.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.94                
2.30                0.00                0.29                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.30                
5.70                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.02                
2.20                1.00                0.29                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.80                0.00                0.29                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.80                0.00                0.29                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.29                
4.80                1.00                0.65                
1.80                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.94                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.29                
5.30                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.29                
4.50                0.00                0.11                
1.30                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.29                
4.80                0.00                0.70                
1.80                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.30                
5.50                1.00                0.71                
1.80                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.30                
5.20                1.00                0.85                
2.00                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.94                
2.90                0.00                0.29                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.94                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.07                
3.00                1.00                0.28                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.94                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.29                
5.30                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.29                
6.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.94                
3.40                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.29                
6.10                1.00                0.93                
2.50                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.29                
5.00                1.00                0.88                
1.50                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.28                
6.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.40                1.00                0.88                
2.10                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.28                
6.30                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.94                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.28                
5.60                1.00                0.93                
2.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.27                
4.80                1.00                0.78                
1.80                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.07                
2.70                1.00                0.27                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.94                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Epoch 5000 RMSE =  0.28058947306919596
Epoch 5100 RMSE =  0.28188531136355943
Epoch 5200 RMSE =  0.2773254487074453
Epoch 5300 RMSE =  0.27912680647986926
Epoch 5400 RMSE =  0.27755571599015644
Epoch 5500 RMSE =  0.2795735321169882
Epoch 5600 RMSE =  0.2814263545157682
Epoch 5700 RMSE =  0.27951030138629157
Epoch 5800 RMSE =  0.277786409266471
Epoch 5900 RMSE =  0.27750858000581097
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.28                
6.70                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.90                0.00                0.27                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
3.80                0.00                0.27                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.27                
4.00                0.00                0.13                
1.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.88                
1.60                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.28                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.40                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.28                
4.70                0.00                0.05                
1.40                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.04                
3.20                1.00                0.28                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.29                
5.70                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.29                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.29                
5.60                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.71                
1.60                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.28                
5.10                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.10                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.28                
4.50                0.00                0.84                
1.50                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.28                
6.10                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.28                
4.50                0.00                0.04                
1.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.29                
6.10                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.40                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.83                
2.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.28                
4.50                0.00                0.20                
1.50                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.29                
5.40                1.00                0.88                
2.10                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.95                
4.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.50                
1.50                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.28                
4.40                0.00                0.48                
1.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.28                
4.80                1.00                0.81                
1.80                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
3.20                0.00                0.27                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.90                
2.00                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.28                
4.50                1.00                0.92                
1.70                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.01                
2.90                1.00                0.27                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.28                
4.80                1.00                0.80                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.28                
5.00                1.00                0.92                
1.50                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.60                1.00                0.27                
4.00                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.02                
2.70                1.00                0.27                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.28                
6.90                1.00                0.95                
2.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.28                
6.00                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.27                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.28                
5.10                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.95                
4.00                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.12                
2.50                1.00                0.27                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.27                
3.90                0.00                0.18                
1.40                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.28                
5.30                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.01                
2.80                1.00                0.27                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.28                
6.30                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.01                
2.30                1.00                0.27                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.28                
6.70                1.00                0.93                
2.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.30                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.95                
2.30                0.00                0.27                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.28                
5.90                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.27                
4.50                0.00                0.01                
1.60                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
2.90                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.27                
5.20                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.50                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                1.00                0.27                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.28                
5.50                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
4.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.27                
5.00                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.27                
3.80                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.27                
5.30                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.27                
6.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.27                
4.80                0.00                0.76                
1.80                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.40                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.27                
4.90                1.00                0.79                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.20                0.00                0.26                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.27                
5.00                1.00                0.91                
1.90                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.60                0.00                0.26                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.01                
2.50                1.00                0.26                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.27                
5.70                1.00                0.90                
2.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.27                
5.60                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.27                
5.40                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.01                
2.90                1.00                0.26                
4.60                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.26                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.00                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.27                
6.40                1.00                0.63                
2.00                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.27                
4.90                1.00                0.88                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                1.00                0.26                
4.00                0.00                0.01                
1.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.27                
4.90                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.40                0.00                0.26                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.27                
4.40                0.00                0.52                
1.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.27                
5.60                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.27                
4.70                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.27                
5.50                1.00                0.91                
2.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.06                
3.00                1.00                0.27                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.60                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.27                
4.60                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.50                0.00                0.27                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.03                
3.00                1.00                0.27                
4.40                0.00                0.00                
1.40                
Epoch 6000 RMSE =  0.2800641530909888
Epoch 6100 RMSE =  0.2783637707871931
Epoch 6200 RMSE =  0.2769349921277673
Epoch 6300 RMSE =  0.2817236734641406
Epoch 6400 RMSE =  0.2803385988707008
Epoch 6500 RMSE =  0.2785015273015197
Epoch 6600 RMSE =  0.2796053075653314
Epoch 6700 RMSE =  0.27653811518566124
Epoch 6800 RMSE =  0.27998232763734493
Epoch 6900 RMSE =  0.2824102857255741
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.29                
6.10                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.27                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.28                
3.90                0.00                0.05                
1.40                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.61                
1.50                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
4.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.27                
4.10                0.00                0.03                
1.30                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
3.00                1.00                0.28                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.50                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.28                
3.80                0.00                0.01                
1.10                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.50                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.29                
6.70                1.00                0.95                
2.00                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.28                
5.10                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.28                
6.10                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.28                
4.40                0.00                0.39                
1.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.28                
5.60                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.60                1.00                0.27                
4.00                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.28                
5.90                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.90                0.00                0.27                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.10                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.28                
6.70                1.00                0.92                
2.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                1.00                0.27                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.27                
4.70                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.02                
2.30                1.00                0.27                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
3.20                0.00                0.27                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.29                
5.40                1.00                0.90                
2.10                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.50                1.00                0.27                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.29                
5.00                1.00                0.90                
1.50                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.04                
2.70                1.00                0.27                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.29                
4.50                1.00                0.92                
1.70                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
2.90                1.00                0.28                
4.60                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.28                
4.00                0.00                0.05                
1.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.01                
2.90                1.00                0.28                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.29                
4.70                0.00                0.08                
1.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.29                
4.60                0.00                0.01                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.30                
5.00                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.30                
6.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.95                
2.30                0.00                0.28                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
3.80                0.00                0.28                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.30                
6.30                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.30                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.29                
4.80                0.00                0.68                
1.80                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
2.90                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.95                
4.00                0.00                0.28                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.29                
4.40                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.09                
3.20                1.00                0.28                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.30                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.20                0.00                0.28                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.30                
6.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.30                
5.30                1.00                0.90                
2.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.29                
4.50                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.30                
6.00                1.00                0.59                
1.80                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.30                
4.90                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.01                
2.80                1.00                0.29                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.30                
5.70                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.29                
4.50                0.00                0.24                
1.50                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.30                
5.00                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.50                0.00                0.28                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.30                
4.80                1.00                0.66                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.30                
5.60                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.29                
5.80                1.00                0.80                
1.60                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.29                
5.50                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.29                
4.50                0.00                0.83                
1.50                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.01                
3.00                1.00                0.28                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.28                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.30                
5.20                1.00                0.89                
2.00                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.30                
5.40                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.25                
2.50                1.00                0.28                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.40                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.29                
4.80                1.00                0.87                
1.80                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.29                
5.50                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.29                
6.40                1.00                0.86                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.29                
5.60                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.29                
5.30                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.95                
4.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.60                0.00                0.27                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.28                
4.90                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.00                
3.20                1.00                0.27                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.60                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.28                
5.70                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.91                
1.60                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.28                
4.50                0.00                0.03                
1.60                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.29                
5.10                1.00                0.88                
2.00                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.29                
4.90                1.00                0.86                
1.80                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.28                
5.10                1.00                0.94                
1.90                
Epoch 7000 RMSE =  0.27711842733509906
Epoch 7100 RMSE =  0.2764202064898198
Epoch 7200 RMSE =  0.2822364545099644
Epoch 7300 RMSE =  0.27765094921765865
Epoch 7400 RMSE =  0.2785963708979998
Epoch 7500 RMSE =  0.28044404797756645
Epoch 7600 RMSE =  0.2799723165156488
Epoch 7700 RMSE =  0.2796988953194952
Epoch 7800 RMSE =  0.28059539578193643
Epoch 7900 RMSE =  0.27998221398777207
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.26                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.50                0.00                0.26                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.27                
5.10                0.00                0.91                
1.60                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.27                
4.50                0.00                0.54                
1.50                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.12                
3.20                1.00                0.27                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.01                
2.90                1.00                0.27                
4.60                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.29                
4.90                1.00                0.73                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.17                
2.50                1.00                0.27                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.28                
4.50                0.00                0.14                
1.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.29                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                1.00                0.28                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.50                1.00                0.28                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.29                
4.00                0.00                0.06                
1.30                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.95                
4.20                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.30                
3.90                0.00                0.07                
1.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.31                
5.30                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.30                
5.10                1.00                0.85                
2.00                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.00                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.30                
5.30                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.30                
4.80                0.00                0.75                
1.80                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.30                
4.50                0.00                0.39                
1.50                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.05                
3.00                1.00                0.29                
4.60                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.31                
5.20                1.00                0.80                
2.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.18                
2.60                1.00                0.29                
4.00                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.31                
5.00                1.00                0.92                
2.00                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.31                
5.60                1.00                0.93                
2.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.01                
2.80                1.00                0.29                
4.70                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
2.90                0.00                0.29                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.31                
6.70                1.00                0.92                
2.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.31                
6.10                1.00                0.91                
2.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.10                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.25                
2.90                1.00                0.29                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.31                
6.60                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.30                
5.20                1.00                0.86                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.30                
5.60                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.30                
4.50                1.00                0.90                
1.70                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.30                
5.10                1.00                0.85                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.60                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.29                
4.70                0.00                0.03                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.30                
5.60                1.00                0.93                
2.40                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.01                
3.00                1.00                0.29                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.30                
4.40                0.00                0.03                
1.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.29                
3.80                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
3.20                0.00                0.29                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.30                0.00                0.29                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.28                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.31                
5.00                1.00                0.89                
1.50                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.28                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.30                
5.00                1.00                0.91                
1.90                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.30                
5.40                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.06                
2.70                1.00                0.28                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.70                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.40                0.00                0.28                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.90                0.00                0.28                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.30                
5.70                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.30                
6.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.50                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.95                
4.00                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.29                
5.10                1.00                0.58                
1.50                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.29                
4.90                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.29                
4.80                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.95                
2.30                0.00                0.27                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.29                
5.50                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.40                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.60                0.00                0.26                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.28                
5.70                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.00                0.00                0.26                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
4.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.26                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.40                0.00                0.26                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.27                
5.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.27                
4.50                0.00                0.13                
1.60                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
3.80                0.00                0.26                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.00                
2.30                1.00                0.26                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.95                
3.70                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.27                
6.00                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.27                
5.50                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.27                
6.30                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.40                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.87                
1.60                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.26                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.27                
5.10                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.27                
4.90                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.20                0.00                0.25                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.27                
4.80                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.27                
6.00                1.00                0.95                
2.50                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.26                
6.40                1.00                0.90                
2.00                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.26                
4.20                0.00                0.04                
1.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.26                
4.40                0.00                0.49                
1.30                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.27                
6.70                1.00                0.91                
2.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.25                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.20                0.00                0.25                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                1.00                0.26                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.27                
5.10                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.01                
3.00                1.00                0.26                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.27                
6.10                1.00                0.93                
2.50                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.27                
5.90                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.26                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.50                1.00                0.87                
1.80                
Epoch 8000 RMSE =  0.2762245526478755
Epoch 8100 RMSE =  0.2761697823526259
Epoch 8200 RMSE =  0.2774146943284646
Epoch 8300 RMSE =  0.2741284387804855
Epoch 8400 RMSE =  0.2787323678459166
Epoch 8500 RMSE =  0.2798662456010352
Epoch 8600 RMSE =  0.2820591291910503
Epoch 8700 RMSE =  0.27509743818635146
Epoch 8800 RMSE =  0.27823758375751845
Epoch 8900 RMSE =  0.2795642448461177
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.26                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.27                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
3.20                0.00                0.26                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                1.00                0.27                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.29                
5.50                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.26                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.68                
1.50                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.28                
5.40                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.40                0.00                0.26                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.28                
6.90                1.00                0.95                
2.30                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
2.90                1.00                0.27                
4.60                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.70                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.10                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.28                
4.70                0.00                0.35                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.29                
5.60                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.50                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.28                
3.90                0.00                0.02                
1.40                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
2.90                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.00                0.00                0.26                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.29                
6.70                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.29                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.81                
2.00                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.40                1.00                0.90                
2.10                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.28                
6.40                1.00                0.77                
2.00                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.10                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.20                0.00                0.26                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.95                
3.70                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.60                1.00                0.27                
4.00                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.26                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.27                
4.00                0.00                0.10                
1.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.40                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.28                
4.90                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.28                
5.00                1.00                0.92                
1.50                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.00                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.70                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.28                
6.30                1.00                0.93                
1.80                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
3.00                1.00                0.26                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.92                
2.00                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.28                
4.90                1.00                0.85                
1.80                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.27                
4.80                1.00                0.88                
1.80                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.20                0.00                0.25                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.25                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.27                
4.50                0.00                0.17                
1.30                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.01                
3.20                1.00                0.26                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.28                
4.50                1.00                0.92                
1.70                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.60                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.27                
4.70                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.95                
4.00                0.00                0.26                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.95                
4.20                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.28                
4.90                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.28                
6.00                1.00                0.95                
2.50                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.25                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.27                
4.50                0.00                0.01                
1.60                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
3.00                1.00                0.26                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.60                0.00                0.26                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.27                
4.60                0.00                0.03                
1.40                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.00                
2.30                1.00                0.27                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.27                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.40                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.29                
5.30                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
3.80                0.00                0.26                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.90                0.00                0.26                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.28                
4.40                0.00                0.19                
1.30                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.29                
4.50                0.00                0.69                
1.50                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.29                
5.10                1.00                0.88                
1.90                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.28                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.30                
6.10                1.00                0.91                
2.50                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.29                
5.60                1.00                0.84                
1.80                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.29                
4.80                0.00                0.17                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.28                
4.40                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.12                
2.50                1.00                0.28                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.30                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.30                
5.50                1.00                0.66                
1.80                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.30                
6.70                1.00                0.90                
2.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.30                
5.20                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.30                
5.90                1.00                0.92                
2.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                1.00                0.28                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.30                
5.00                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.30                
6.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.29                
5.30                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.29                
5.70                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.29                
5.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.29                
6.10                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.08                
2.70                1.00                0.27                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.40                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.29                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.26                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.28                
6.00                1.00                0.83                
1.80                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.21                
2.50                1.00                0.27                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.29                
5.50                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.95                
2.30                0.00                0.26                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.27                
3.80                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.29                
5.00                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.90                1.00                0.27                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.29                
5.10                1.00                0.90                
1.80                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.29                
5.70                1.00                0.93                
2.10                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.29                
5.80                1.00                0.82                
1.60                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.29                
5.60                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.28                
4.80                1.00                0.86                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
4.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.50                0.00                0.26                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.28                
5.10                0.00                0.90                
1.60                
Epoch 9000 RMSE =  0.2726152301304075
Epoch 9100 RMSE =  0.2766572762987714
Epoch 9200 RMSE =  0.27960156359461363
Epoch 9300 RMSE =  0.27645847714803756
Epoch 9400 RMSE =  0.27543873611160896
Epoch 9500 RMSE =  0.2781714439276821
Epoch 9600 RMSE =  0.27524014707185046
Epoch 9700 RMSE =  0.27518289524876444
Epoch 9800 RMSE =  0.2758602502616267
Epoch 9900 RMSE =  0.27833111778042163
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.28                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.30                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.29                
5.10                1.00                0.90                
2.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.01                
2.80                1.00                0.27                
4.00                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
7.90                0.00                0.00                
3.80                0.00                0.29                
6.40                1.00                0.68                
2.00                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
3.80                0.00                0.27                
1.70                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.40                0.00                0.29                
5.60                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.90                1.00                0.28                
4.70                0.00                0.22                
1.40                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.40                1.00                0.28                
3.80                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.20                1.00                0.29                
4.80                0.00                0.68                
1.80                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.50                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                0.00                0.29                
5.00                1.00                0.78                
1.50                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.80                1.00                0.95                
4.00                0.00                0.27                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.00                0.00                0.29                
5.50                1.00                0.87                
2.10                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.09                
3.00                1.00                0.27                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.70                0.00                0.29                
4.90                1.00                0.72                
1.80                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                1.00                0.28                
4.50                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.60                0.00                0.30                
6.90                1.00                0.95                
2.30                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.60                0.00                0.27                
1.00                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.70                1.00                0.29                
5.10                0.00                0.82                
1.60                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.29                
5.50                1.00                0.73                
1.80                
Input Values        Expected Values     Predicted Values    
5.70                1.00                0.95                
4.40                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.20                0.00                0.29                
5.30                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.60                0.00                0.29                
6.10                1.00                0.93                
2.50                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
2.90                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.01                
2.50                1.00                0.27                
3.90                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.40                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.00                0.00                0.29                
5.80                1.00                0.47                
1.60                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.70                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                0.00                0.29                
4.90                1.00                0.91                
1.80                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.29                
5.70                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.95                
2.20                
Input Values        Expected Values     Predicted Values    
5.10                0.00                0.01                
2.50                1.00                0.27                
3.00                0.00                0.00                
1.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.50                0.00                0.29                
5.00                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
4.50                1.00                0.95                
2.30                0.00                0.26                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
7.60                0.00                0.00                
3.00                0.00                0.28                
6.60                1.00                0.95                
2.10                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.00                
2.50                0.00                0.28                
4.50                1.00                0.94                
1.70                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.40                1.00                0.28                
4.50                0.00                0.17                
1.60                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.40                0.00                0.26                
1.50                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
6.60                0.00                0.00                
2.90                1.00                0.28                
4.60                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.60                1.00                0.28                
4.00                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.90                0.00                0.27                
1.70                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
7.30                0.00                0.00                
2.90                0.00                0.29                
6.30                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
2.80                0.00                0.29                
6.70                1.00                0.94                
2.00                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.28                
4.50                0.00                0.56                
1.50                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
3.20                0.00                0.27                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
3.00                1.00                0.28                
4.60                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
3.40                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.80                1.00                0.28                
4.70                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                0.00                0.29                
5.00                1.00                0.91                
1.90                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.00                0.00                0.29                
6.10                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.80                0.00                0.29                
4.80                1.00                0.74                
1.80                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.30                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.60                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                0.00                0.29                
6.00                1.00                0.94                
2.50                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.80                0.00                0.28                
4.90                1.00                0.93                
2.00                
Input Values        Expected Values     Predicted Values    
7.70                0.00                0.00                
3.80                0.00                0.28                
6.70                1.00                0.92                
2.20                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.90                1.00                0.27                
4.20                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.40                0.00                0.26                
1.60                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.80                0.00                0.28                
5.60                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.90                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.90                0.00                0.28                
5.60                1.00                0.92                
1.80                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.20                1.00                0.28                
4.50                0.00                0.74                
1.50                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.26                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
3.10                0.00                0.28                
5.50                1.00                0.73                
1.80                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.60                1.00                0.27                
4.40                0.00                0.01                
1.20                
Input Values        Expected Values     Predicted Values    
7.00                0.00                0.07                
3.20                1.00                0.27                
4.70                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.29                
5.10                1.00                0.92                
1.90                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.40                0.00                0.26                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                0.00                0.28                
5.20                1.00                0.92                
2.30                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.02                
2.90                1.00                0.27                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
3.00                0.00                0.28                
4.80                1.00                0.68                
1.80                
Input Values        Expected Values     Predicted Values    
5.20                0.00                0.00                
2.70                1.00                0.28                
3.90                0.00                0.06                
1.40                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.95                
4.20                0.00                0.27                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.50                0.00                0.27                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.20                0.00                0.26                
1.20                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.50                0.00                0.26                
1.60                0.00                0.00                
0.60                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
2.50                0.00                0.28                
5.80                1.00                0.94                
1.80                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
3.40                0.00                0.28                
5.40                1.00                0.93                
2.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.26                
1.50                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.01                
2.70                1.00                0.27                
4.10                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                0.00                0.28                
5.10                1.00                0.89                
1.80                
Input Values        Expected Values     Predicted Values    
7.10                0.00                0.00                
3.00                0.00                0.28                
5.90                1.00                0.94                
2.10                
Input Values        Expected Values     Predicted Values    
6.40                0.00                0.00                
2.70                0.00                0.28                
5.30                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.20                0.00                0.28                
5.10                1.00                0.87                
2.00                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.80                0.00                0.27                
5.10                1.00                0.74                
1.50                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
3.00                0.00                0.27                
5.80                1.00                0.94                
2.20                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.20                1.00                0.26                
4.00                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.00                0.00                0.26                
1.40                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.00                0.00                0.26                
1.40                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
3.00                1.00                0.27                
4.20                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.20                0.00                0.28                
5.70                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.30                1.00                0.27                
4.40                0.00                0.39                
1.30                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.00                
2.80                1.00                0.27                
4.50                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.80                0.00                0.28                
5.10                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.50                1.00                0.27                
4.00                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
7.20                0.00                0.00                
3.20                0.00                0.28                
6.00                1.00                0.82                
1.80                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                0.00                0.28                
5.40                1.00                0.90                
2.10                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.01                
2.30                1.00                0.27                
3.30                0.00                0.00                
1.00                
Epoch 10000 RMSE =  0.27906759253347235
Final Training Epoch RMSE =  0.27906759253347235
Input Values        Expected Values     Predicted Values    
4.70                1.00                0.95                
3.20                0.00                0.27                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.60                1.00                0.95                
3.40                0.00                0.27                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.40                0.00                0.27                
1.50                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.90                1.00                0.95                
3.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
4.30                1.00                0.95                
3.00                0.00                0.27                
1.10                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.90                0.00                0.27                
1.30                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
5.40                1.00                0.95                
3.40                0.00                0.27                
1.70                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.30                0.00                0.27                
1.70                0.00                0.00                
0.50                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.40                0.00                0.27                
1.90                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.00                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.70                1.00                0.95                
3.20                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.10                0.00                0.27                
1.60                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.20                1.00                0.95                
4.10                0.00                0.27                
1.50                0.00                0.00                
0.10                
Input Values        Expected Values     Predicted Values    
5.50                1.00                0.95                
3.50                0.00                0.27                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
4.40                1.00                0.95                
3.00                0.00                0.27                
1.30                0.00                0.00                
0.20                
Input Values        Expected Values     Predicted Values    
5.00                1.00                0.95                
3.50                0.00                0.27                
1.30                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
5.10                1.00                0.95                
3.80                0.00                0.27                
1.90                0.00                0.00                
0.40                
Input Values        Expected Values     Predicted Values    
4.80                1.00                0.95                
3.00                0.00                0.27                
1.40                0.00                0.00                
0.30                
Input Values        Expected Values     Predicted Values    
6.90                0.00                0.00                
3.10                1.00                0.28                
4.90                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.00                
2.30                1.00                0.28                
4.00                0.00                0.10                
1.30                
Input Values        Expected Values     Predicted Values    
6.50                0.00                0.00                
2.80                1.00                0.28                
4.60                0.00                0.03                
1.50                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
3.30                1.00                0.28                
4.70                0.00                0.01                
1.60                
Input Values        Expected Values     Predicted Values    
4.90                0.00                0.02                
2.40                1.00                0.27                
3.30                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.00                0.00                0.00                
2.00                1.00                0.28                
3.50                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.90                0.00                0.00                
3.00                1.00                0.28                
4.20                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.05                
2.90                1.00                0.27                
3.60                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.02                
3.10                1.00                0.27                
4.40                0.00                0.00                
1.40                
Input Values        Expected Values     Predicted Values    
6.30                0.00                0.00                
2.50                1.00                0.28                
4.90                0.00                0.70                
1.50                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
2.80                1.00                0.28                
4.80                0.00                0.01                
1.40                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.00                1.00                0.28                
5.00                0.00                0.36                
1.70                
Input Values        Expected Values     Predicted Values    
6.00                0.00                0.00                
2.90                1.00                0.28                
4.50                0.00                0.05                
1.50                
Input Values        Expected Values     Predicted Values    
5.70                0.00                0.43                
2.60                1.00                0.27                
3.50                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.50                0.00                0.01                
2.40                1.00                0.27                
3.70                0.00                0.00                
1.00                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.01                
2.70                1.00                0.27                
3.90                0.00                0.00                
1.20                
Input Values        Expected Values     Predicted Values    
5.40                0.00                0.00                
3.00                1.00                0.28                
4.50                0.00                0.23                
1.50                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.10                1.00                0.28                
4.70                0.00                0.00                
1.50                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
3.00                1.00                0.27                
4.10                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.60                0.00                0.00                
2.70                1.00                0.28                
4.20                0.00                0.01                
1.30                
Input Values        Expected Values     Predicted Values    
6.20                0.00                0.00                
2.90                1.00                0.27                
4.30                0.00                0.00                
1.30                
Input Values        Expected Values     Predicted Values    
5.80                0.00                0.00                
2.70                0.00                0.29                
5.10                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
7.40                0.00                0.00                
2.80                0.00                0.29                
6.10                1.00                0.93                
1.90                
Input Values        Expected Values     Predicted Values    
6.10                0.00                0.00                
2.60                0.00                0.28                
5.60                1.00                0.89                
1.40                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.10                0.00                0.29                
5.60                1.00                0.94                
2.40                
Input Values        Expected Values     Predicted Values    
6.80                0.00                0.00                
3.20                0.00                0.29                
5.90                1.00                0.94                
2.30                
Input Values        Expected Values     Predicted Values    
6.70                0.00                0.00                
3.30                0.00                0.29                
5.70                1.00                0.94                
2.50                
Test Epoch RMSE =  0.3184761248673846
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.77                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.74                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.78                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.77                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.70                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.76                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.77                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.73                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.73                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.72                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.77                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.77                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.73                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.74                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.78                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.75                
Epoch 0 RMSE =  0.2574563110188265
Epoch 100 RMSE =  0.25447759775078765
Epoch 200 RMSE =  0.25273350890086604
Epoch 300 RMSE =  0.2514490931798019
Epoch 400 RMSE =  0.2504342263959059
Epoch 500 RMSE =  0.24957761976742207
Epoch 600 RMSE =  0.24879305195777626
Epoch 700 RMSE =  0.24800679055817684
Epoch 800 RMSE =  0.24711712939561814
Epoch 900 RMSE =  0.24597731205374887
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.76                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.76                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.70                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.69                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.71                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.76                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.71                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.72                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.73                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.75                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.76                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.77                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.77                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.77                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.65                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.74                
Epoch 1000 RMSE =  0.2443408489004246
Epoch 1100 RMSE =  0.2417156427306679
Epoch 1200 RMSE =  0.23718509484077657
Epoch 1300 RMSE =  0.22933320771789625
Epoch 1400 RMSE =  0.21711685909296694
Epoch 1500 RMSE =  0.20163089554137265
Epoch 1600 RMSE =  0.1855405872370659
Epoch 1700 RMSE =  0.1707408089206214
Epoch 1800 RMSE =  0.1578374744249339
Epoch 1900 RMSE =  0.14675919224996054
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.82                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.53                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.84                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.61                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.82                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.82                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.81                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.58                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.71                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.33                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.76                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.67                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.62                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.84                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.78                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.84                
Epoch 2000 RMSE =  0.13723612901196847
Epoch 2100 RMSE =  0.12898506837939738
Epoch 2200 RMSE =  0.12176537124732263
Epoch 2300 RMSE =  0.11538423957993896
Epoch 2400 RMSE =  0.10969793307504198
Epoch 2500 RMSE =  0.10458897089624973
Epoch 2600 RMSE =  0.09996836518873695
Epoch 2700 RMSE =  0.09576938503342515
Epoch 2800 RMSE =  0.0919311854650668
Epoch 2900 RMSE =  0.08841104475968302
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.71                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.86                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.88                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.58                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.88                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.18                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.78                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.84                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.52                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.66                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.57                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.85                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.88                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.86                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.44                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.81                
Epoch 3000 RMSE =  0.08517022064630188
Epoch 3100 RMSE =  0.0821772103418182
Epoch 3200 RMSE =  0.0794055731401022
Epoch 3300 RMSE =  0.07683263908994833
Epoch 3400 RMSE =  0.07443897255439383
Epoch 3500 RMSE =  0.07220792139256327
Epoch 3600 RMSE =  0.07012383497303704
Epoch 3700 RMSE =  0.06817473689234205
Epoch 3800 RMSE =  0.06634774284764113
Epoch 3900 RMSE =  0.06463466594639751
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.90                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.55                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.90                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.49                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.86                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.88                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.13                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.82                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.79                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.65                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.87                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.56                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.88                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.40                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.90                
Epoch 4000 RMSE =  0.06302489793187778
Epoch 4100 RMSE =  0.06151044001408949
Epoch 4200 RMSE =  0.060084814580504436
Epoch 4300 RMSE =  0.05874102764551198
Epoch 4400 RMSE =  0.05747280971043549
Epoch 4500 RMSE =  0.05627559848329719
Epoch 4600 RMSE =  0.05514382449724477
Epoch 4700 RMSE =  0.05407306452344606
Epoch 4800 RMSE =  0.053059582378421725
Epoch 4900 RMSE =  0.052099853303160096
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.54                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.38                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.91                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.11                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.80                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.91                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.87                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.83                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.55                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.91                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.89                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.89                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.65                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.89                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.47                
Epoch 5000 RMSE =  0.051189667735855114
Epoch 5100 RMSE =  0.05032643637524879
Epoch 5200 RMSE =  0.04950664827655408
Epoch 5300 RMSE =  0.04872870602966698
Epoch 5400 RMSE =  0.047988971435588945
Epoch 5500 RMSE =  0.047286318666940476
Epoch 5600 RMSE =  0.04661754770589209
Epoch 5700 RMSE =  0.0459809619225428
Epoch 5800 RMSE =  0.04537484199082082
Epoch 5900 RMSE =  0.04479759984937088
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.90                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.53                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.90                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.92                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.36                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.84                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.92                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.92                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.89                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.81                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.10                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.46                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.88                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.65                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.54                
Epoch 6000 RMSE =  0.044247317081099975
Epoch 6100 RMSE =  0.04372264185101898
Epoch 6200 RMSE =  0.04322231914133063
Epoch 6300 RMSE =  0.04274478642765219
Epoch 6400 RMSE =  0.04228866574155293
Epoch 6500 RMSE =  0.04185302397180722
Epoch 6600 RMSE =  0.04143696179022491
Epoch 6700 RMSE =  0.04103962647629019
Epoch 6800 RMSE =  0.04065902319929909
Epoch 6900 RMSE =  0.040295584106224934
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.52                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.45                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.91                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.93                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.92                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.65                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.81                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.93                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.53                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.84                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.89                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.91                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.90                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.35                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Epoch 7000 RMSE =  0.039947639541020166
Epoch 7100 RMSE =  0.039614336557350825
Epoch 7200 RMSE =  0.039295201089870176
Epoch 7300 RMSE =  0.03898946526141678
Epoch 7400 RMSE =  0.03869585376575042
Epoch 7500 RMSE =  0.03841531097774525
Epoch 7600 RMSE =  0.038145970902750743
Epoch 7700 RMSE =  0.037887392267374226
Epoch 7800 RMSE =  0.03763920477882769
Epoch 7900 RMSE =  0.03740095361927906
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.85                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.89                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.90                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.93                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.73                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.93                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.53                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.34                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.44                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.65                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.52                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.91                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.91                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.93                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.08                
Epoch 8000 RMSE =  0.0371720955382213
Epoch 8100 RMSE =  0.0369521110723613
Epoch 8200 RMSE =  0.03674066279964945
Epoch 8300 RMSE =  0.036537350180097614
Epoch 8400 RMSE =  0.036341621316899574
Epoch 8500 RMSE =  0.03615353829968959
Epoch 8600 RMSE =  0.03597237614901165
Epoch 8700 RMSE =  0.0357976456490986
Epoch 8800 RMSE =  0.03562900862580513
Epoch 8900 RMSE =  0.03546739906552532
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.44                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.85                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.91                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.53                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.92                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.34                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.93                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.93                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.65                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.52                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.92                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.73                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.90                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.08                
Epoch 9000 RMSE =  0.03531111214001809
Epoch 9100 RMSE =  0.03516051461826725
Epoch 9200 RMSE =  0.035015122355246536
Epoch 9300 RMSE =  0.03487471491972499
Epoch 9400 RMSE =  0.03473912322125052
Epoch 9500 RMSE =  0.03460831156005401
Epoch 9600 RMSE =  0.03448132469602433
Epoch 9700 RMSE =  0.034358893613416576
Epoch 9800 RMSE =  0.03424099805629527
Epoch 9900 RMSE =  0.03412667908406183
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.64                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.92                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.92                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.90                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.52                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.85                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.94                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.73                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.94                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.91                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.08                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.34                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.51                
Epoch 10000 RMSE =  0.03401588300688503
Final Training Epoch RMSE =  0.03401588300688503
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.07                
Input Values        Expected Values     Predicted Values    
0.01                0.01                0.08                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.08                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.09                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.09                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.10                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.10                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.10                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.11                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.12                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.12                
Input Values        Expected Values     Predicted Values    
0.12                0.12                0.13                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.13                
Input Values        Expected Values     Predicted Values    
0.14                0.14                0.14                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.14                
Input Values        Expected Values     Predicted Values    
0.16                0.16                0.15                
Input Values        Expected Values     Predicted Values    
0.17                0.17                0.16                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.16                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.17                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.18                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.19                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.19                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.20                
Input Values        Expected Values     Predicted Values    
0.24                0.24                0.21                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.22                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.29                0.29                0.25                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.26                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.27                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.28                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.29                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.30                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
0.36                0.35                0.32                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.35                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.36                
Input Values        Expected Values     Predicted Values    
0.40                0.39                0.37                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.38                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.39                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.40                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.45                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.46                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.47                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.48                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.49                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.50                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.53                
Input Values        Expected Values     Predicted Values    
0.56                0.53                0.54                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.56                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.57                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.58                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.59                
Input Values        Expected Values     Predicted Values    
0.61                0.57                0.60                
Input Values        Expected Values     Predicted Values    
0.62                0.58                0.61                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.62                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.63                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.67                0.62                0.65                
Input Values        Expected Values     Predicted Values    
0.68                0.63                0.66                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.68                
Input Values        Expected Values     Predicted Values    
0.71                0.65                0.69                
Input Values        Expected Values     Predicted Values    
0.72                0.66                0.70                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.71                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.72                
Input Values        Expected Values     Predicted Values    
0.77                0.70                0.73                
Input Values        Expected Values     Predicted Values    
0.78                0.70                0.74                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.75                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.76                
Input Values        Expected Values     Predicted Values    
0.82                0.73                0.77                
Input Values        Expected Values     Predicted Values    
0.83                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.78                
Input Values        Expected Values     Predicted Values    
0.85                0.75                0.78                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.80                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
0.89                0.78                0.81                
Input Values        Expected Values     Predicted Values    
0.90                0.78                0.81                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.82                
Input Values        Expected Values     Predicted Values    
0.93                0.80                0.83                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.95                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.97                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.99                0.84                0.85                
Input Values        Expected Values     Predicted Values    
1.01                0.85                0.86                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.86                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.87                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.87                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.09                0.89                0.88                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.88                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.89                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.89                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.89                
Input Values        Expected Values     Predicted Values    
1.14                0.91                0.89                
Input Values        Expected Values     Predicted Values    
1.15                0.91                0.90                
Input Values        Expected Values     Predicted Values    
1.17                0.92                0.90                
Input Values        Expected Values     Predicted Values    
1.18                0.92                0.90                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.91                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.91                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.91                
Input Values        Expected Values     Predicted Values    
1.23                0.94                0.91                
Input Values        Expected Values     Predicted Values    
1.24                0.95                0.91                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.92                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.92                
Input Values        Expected Values     Predicted Values    
1.29                0.96                0.92                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.92                
Input Values        Expected Values     Predicted Values    
1.31                0.97                0.93                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.93                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.93                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.93                
Input Values        Expected Values     Predicted Values    
1.35                0.98                0.93                
Input Values        Expected Values     Predicted Values    
1.36                0.98                0.93                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.93                
Input Values        Expected Values     Predicted Values    
1.38                0.98                0.93                
Input Values        Expected Values     Predicted Values    
1.40                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.48                1.00                0.94                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.94                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.54                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.95                
Test Epoch RMSE =  0.03159509025470466
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.67                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.72                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.75                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.71                
0.00                
Epoch 0 RMSE =  0.5417115399828347
Epoch 100 RMSE =  0.5040378966208318
Epoch 200 RMSE =  0.5012508589539905
Epoch 300 RMSE =  0.5010650061075899
Epoch 400 RMSE =  0.5010040653639926
Epoch 500 RMSE =  0.500953757534275
Epoch 600 RMSE =  0.5008957697077202
Epoch 700 RMSE =  0.500845721469074
Epoch 800 RMSE =  0.5007921396146013
Epoch 900 RMSE =  0.5007351210027647
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.50                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.50                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.51                
1.00                
Epoch 1000 RMSE =  0.5006853059932042
Epoch 1100 RMSE =  0.5006265887090119
Epoch 1200 RMSE =  0.5005747769252588
Epoch 1300 RMSE =  0.5005173118723265
Epoch 1400 RMSE =  0.500455938993132
Epoch 1500 RMSE =  0.5004002358984155
Epoch 1600 RMSE =  0.5003383733324602
Epoch 1700 RMSE =  0.5002732577584128
Epoch 1800 RMSE =  0.5002038864329651
Epoch 1900 RMSE =  0.5001334200082088
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.50                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.50                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.51                
1.00                
Epoch 2000 RMSE =  0.500064482857816
Epoch 2100 RMSE =  0.499983647486612
Epoch 2200 RMSE =  0.49990513798416913
Epoch 2300 RMSE =  0.49981831193100607
Epoch 2400 RMSE =  0.4997273992850105
Epoch 2500 RMSE =  0.4996247979493404
Epoch 2600 RMSE =  0.4995233746962609
Epoch 2700 RMSE =  0.49940672103121064
Epoch 2800 RMSE =  0.4992882680136648
Epoch 2900 RMSE =  0.49915765607259216
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.49                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.52                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.50                
0.00                
Epoch 3000 RMSE =  0.49901485461193756
Epoch 3100 RMSE =  0.4988592384731587
Epoch 3200 RMSE =  0.49868543579904795
Epoch 3300 RMSE =  0.4985011741605699
Epoch 3400 RMSE =  0.4983007996428
Epoch 3500 RMSE =  0.498077700788499
Epoch 3600 RMSE =  0.4978282475663385
Epoch 3700 RMSE =  0.49756005680682436
Epoch 3800 RMSE =  0.4972531490280827
Epoch 3900 RMSE =  0.49691831502952366
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.51                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.48                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.53                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.52                
1.00                
Epoch 4000 RMSE =  0.49655115382277865
Epoch 4100 RMSE =  0.49613703991391195
Epoch 4200 RMSE =  0.4956759492402786
Epoch 4300 RMSE =  0.49516202148888877
Epoch 4400 RMSE =  0.49458910910312637
Epoch 4500 RMSE =  0.49395033199023547
Epoch 4600 RMSE =  0.49323852650528893
Epoch 4700 RMSE =  0.4924501725521626
Epoch 4800 RMSE =  0.4915740403982774
Epoch 4900 RMSE =  0.49060374344536
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.54                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.47                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.51                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.53                
1.00                
Epoch 5000 RMSE =  0.4895314878230768
Epoch 5100 RMSE =  0.48836267350471013
Epoch 5200 RMSE =  0.48707745253960527
Epoch 5300 RMSE =  0.4856738605702414
Epoch 5400 RMSE =  0.4841611552131758
Epoch 5500 RMSE =  0.4825243617845885
Epoch 5600 RMSE =  0.48077224388812023
Epoch 5700 RMSE =  0.4788942592493785
Epoch 5800 RMSE =  0.4769088979616131
Epoch 5900 RMSE =  0.47481143801603626
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.45                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.56                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.53                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.53                
0.00                
Epoch 6000 RMSE =  0.47260161796905487
Epoch 6100 RMSE =  0.4702863680555335
Epoch 6200 RMSE =  0.4678774829121297
Epoch 6300 RMSE =  0.4653686193155048
Epoch 6400 RMSE =  0.4627875504818511
Epoch 6500 RMSE =  0.46012378366493684
Epoch 6600 RMSE =  0.4573891341616715
Epoch 6700 RMSE =  0.45459157404789513
Epoch 6800 RMSE =  0.45174386305131004
Epoch 6900 RMSE =  0.44884244778861265
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.56                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.41                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.58                
1.00                
Epoch 7000 RMSE =  0.4459184524424392
Epoch 7100 RMSE =  0.4429613087392056
Epoch 7200 RMSE =  0.43997332666772554
Epoch 7300 RMSE =  0.43699141834005506
Epoch 7400 RMSE =  0.4340013992746685
Epoch 7500 RMSE =  0.4310043810523513
Epoch 7600 RMSE =  0.42803655385875694
Epoch 7700 RMSE =  0.42507815013714567
Epoch 7800 RMSE =  0.42214061113010143
Epoch 7900 RMSE =  0.4192253782348416
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.61                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.60                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.49                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.37                
0.00                
Epoch 8000 RMSE =  0.4163354542705994
Epoch 8100 RMSE =  0.4134883605780635
Epoch 8200 RMSE =  0.41065930858125266
Epoch 8300 RMSE =  0.40787409169750816
Epoch 8400 RMSE =  0.4051248366020977
Epoch 8500 RMSE =  0.40239566970598245
Epoch 8600 RMSE =  0.3997038817144644
Epoch 8700 RMSE =  0.3970553960259622
Epoch 8800 RMSE =  0.3944256691145184
Epoch 8900 RMSE =  0.3918377559482846
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.63                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.33                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.63                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.47                
1.00                
Epoch 9000 RMSE =  0.3892697417325994
Epoch 9100 RMSE =  0.3867389161055763
Epoch 9200 RMSE =  0.38423736705684364
Epoch 9300 RMSE =  0.38176009274860084
Epoch 9400 RMSE =  0.37932713123752304
Epoch 9500 RMSE =  0.3769123232558209
Epoch 9600 RMSE =  0.37451608327935865
Epoch 9700 RMSE =  0.3721640413587679
Epoch 9800 RMSE =  0.36982925362097374
Epoch 9900 RMSE =  0.36751917633886266
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.65                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.65                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.30                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.45                
1.00                
Epoch 10000 RMSE =  0.3652207753794612
Epoch 10100 RMSE =  0.36295617112617645
Epoch 10200 RMSE =  0.3607069699310952
Epoch 10300 RMSE =  0.3584808494290038
Epoch 10400 RMSE =  0.35625954366128043
Epoch 10500 RMSE =  0.35405631747672234
Epoch 10600 RMSE =  0.35186010919391386
Epoch 10700 RMSE =  0.3496488517159203
Epoch 10800 RMSE =  0.3474390958735072
Epoch 10900 RMSE =  0.34521908656148687
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.67                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.27                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.42                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.66                
1.00                
Epoch 11000 RMSE =  0.3429687114619973
Epoch 11100 RMSE =  0.3406685607900305
Epoch 11200 RMSE =  0.33832560283584934
Epoch 11300 RMSE =  0.33591314295871727
Epoch 11400 RMSE =  0.33339856838183457
Epoch 11500 RMSE =  0.33075846527332675
Epoch 11600 RMSE =  0.3279537194301096
Epoch 11700 RMSE =  0.3249640859274432
Epoch 11800 RMSE =  0.32173897842904503
Epoch 11900 RMSE =  0.31823916339164354
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.70                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.38                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.25                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.69                
1.00                
Epoch 12000 RMSE =  0.3144143793064916
Epoch 12100 RMSE =  0.3102038342513589
Epoch 12200 RMSE =  0.30560548224321676
Epoch 12300 RMSE =  0.300550028333377
Epoch 12400 RMSE =  0.2950599565150433
Epoch 12500 RMSE =  0.2891326591785841
Epoch 12600 RMSE =  0.2827877386178084
Epoch 12700 RMSE =  0.27610332562820405
Epoch 12800 RMSE =  0.2691333650016055
Epoch 12900 RMSE =  0.2619880504420729
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.74                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.22                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.29                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.76                
0.00                
Epoch 13000 RMSE =  0.25474710803121803
Epoch 13100 RMSE =  0.24750651075677124
Epoch 13200 RMSE =  0.24035575833292708
Epoch 13300 RMSE =  0.23336432311065897
Epoch 13400 RMSE =  0.22658704742534283
Epoch 13500 RMSE =  0.22006498740271857
Epoch 13600 RMSE =  0.213825194031944
Epoch 13700 RMSE =  0.20788642113337494
Epoch 13800 RMSE =  0.20225010012832
Epoch 13900 RMSE =  0.19691719964745696
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.80                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.20                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.83                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.19                
0.00                
Epoch 14000 RMSE =  0.19187980944184121
Epoch 14100 RMSE =  0.187125917637491
Epoch 14200 RMSE =  0.18264307941636798
Epoch 14300 RMSE =  0.1784161536445553
Epoch 14400 RMSE =  0.17442879811344833
Epoch 14500 RMSE =  0.17066733953655958
Epoch 14600 RMSE =  0.16711612089793135
Epoch 14700 RMSE =  0.16376026448275785
Epoch 14800 RMSE =  0.16058665236070058
Epoch 14900 RMSE =  0.1575818843513448
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.84                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.86                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.17                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.15                
1.00                
Epoch 15000 RMSE =  0.1547334183018709
Epoch 15100 RMSE =  0.15203094512660384
Epoch 15200 RMSE =  0.14946354961128516
Epoch 15300 RMSE =  0.14702185401994297
Epoch 15400 RMSE =  0.14469695002426317
Epoch 15500 RMSE =  0.14248075124557344
Epoch 15600 RMSE =  0.1403656665531118
Epoch 15700 RMSE =  0.13834532016442583
Epoch 15800 RMSE =  0.136413085994539
Epoch 15900 RMSE =  0.13456309804965083
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.12                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.87                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.15                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.88                
0.00                
Epoch 16000 RMSE =  0.13279040825158522
Epoch 16100 RMSE =  0.13108966497316601
Epoch 16200 RMSE =  0.1294565050413379
Epoch 16300 RMSE =  0.1278867093468701
Epoch 16400 RMSE =  0.12637666006166398
Epoch 16500 RMSE =  0.12492283725388102
Epoch 16600 RMSE =  0.12352165525211961
Epoch 16700 RMSE =  0.12217036609380942
Epoch 16800 RMSE =  0.1208659924590145
Epoch 16900 RMSE =  0.11960595321251537
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.89                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.14                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.10                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.88                
1.00                
Epoch 17000 RMSE =  0.11838802678337727
Epoch 17100 RMSE =  0.11720980999381334
Epoch 17200 RMSE =  0.1160690635561413
Epoch 17300 RMSE =  0.11496426357592658
Epoch 17400 RMSE =  0.11389325132738487
Epoch 17500 RMSE =  0.11285447164076703
Epoch 17600 RMSE =  0.11184631825274181
Epoch 17700 RMSE =  0.11086726131574046
Epoch 17800 RMSE =  0.10991608975900204
Epoch 17900 RMSE =  0.10899132155779136
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.90                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.13                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.09                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.89                
1.00                
Epoch 18000 RMSE =  0.10809184326139364
Epoch 18100 RMSE =  0.10721648161620116
Epoch 18200 RMSE =  0.10636427369099662
Epoch 18300 RMSE =  0.10553404712135886
Epoch 18400 RMSE =  0.10472496222190726
Epoch 18500 RMSE =  0.10393607112205004
Epoch 18600 RMSE =  0.10316664000597357
Epoch 18700 RMSE =  0.10241575927811655
Epoch 18800 RMSE =  0.10168270816524894
Epoch 18900 RMSE =  0.10096676738203604
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.13                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.08                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.90                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.91                
0.00                
Epoch 19000 RMSE =  0.10026730799124517
Epoch 19100 RMSE =  0.09958373189593027
Epoch 19200 RMSE =  0.09891529817507716
Epoch 19300 RMSE =  0.09826159767458115
Epoch 19400 RMSE =  0.09762198068447073
Epoch 19500 RMSE =  0.09699596746854192
Epoch 19600 RMSE =  0.09638306689247253
Epoch 19700 RMSE =  0.09578282875648597
Epoch 19800 RMSE =  0.0951948025390128
Epoch 19900 RMSE =  0.0946185486066479
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.91                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.12                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.07                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.91                
1.00                
Epoch 20000 RMSE =  0.09405371679042474
Final Training Epoch RMSE =  0.09405371679042474
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.12                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.91                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.91                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.07                
1.00                
Test Epoch RMSE =  0.09402046517834951

Process finished with exit code 0

"""
