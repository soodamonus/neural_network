from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import random
from collections import deque
import math
import json

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
            _features: numpy array containing features in a NNdataset
            _labels: numpy array containing labels in a NNdataset
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


class MultiTypeEncoder(json.JSONEncoder):
    """Class to convert NNData, NDarray, and deque datatypes to JSON type."""
    def default(self, o):
        if isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        if isinstance(o, np.ndarray):
            return {"__NDarray__": list(o)}
        if isinstance(o, deque):
            return {"__deque__": list(o)}
        else:
            super().default(o)


def multi_type_decoder(o):
    """ Takes JSON datatype and converts back into NNData object, attributes
    inclusive."""
    if "__NNData__" in o:
        # isolate the object into item
        item = o["__NNData__"]
        # turn item back into object
        ret_obj = NNData(item["_features"], item["_labels"],
                                     item["_train_factor"])
        ret_obj._train_indices = item["_train_indices"]
        ret_obj._test_indices = item["_test_indices"]
        ret_obj._train_pool = item["_train_pool"]
        ret_obj._test_pool = item["_test_pool"]
        return ret_obj
    if "__NDarray__" in o:
        return np.array(o["__NDarray__"])
    if "__deque__" in o:
        return deque(o["__deque__"])
    else:
        return o


def run_xor_decoded(features, labels):
    """Test and train XOR data on NN model"""
    network = FFBPNetwork(2, 1)
    # add a hidden layer of 3 neurodes
    network.add_hidden_layer(3)
    # train on all of the data
    train_factor = 1
    data = NNData(features, labels, train_factor)
    x, y = network.train(data, 20001, order=NNData.Order.RANDOM)
    data.split_set(0)
    data.prime_data()
    network.test(data)
    return x, y


def run_sin_decoded(features, labels):
    """Test and train sine function from 0 radians to 1.57 radians."""
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(5)
    sin_X = features
    sin_Y = labels
    data = NNData(sin_X, sin_Y, 0.7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    x, y = network.test(data)
    return x, y


def main():
    """
    Encode XOR data from JSON datatype, decode it, and test through NN model.
    """
    # load XOR data into a NNData object called xor_data
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    xor_data = NNData(features, labels, 1)
    # JSON Encode xor_data into an object xor_data_encoded
    xor_data_encoded = json.dumps(xor_data, cls=MultiTypeEncoder)
    # Decode xor_data_encoded into an NNData object called xor_data_decoded
    xor_data_decoded = json.loads(xor_data_encoded,
                                  object_hook=multi_type_decoder)
    run_xor_decoded(xor_data_decoded._features, xor_data_decoded._labels)
    """
    Decode JSON sin data from external .txt file and test through NN model.
    """
    # open encoded sin_data from file
    with open("sin_data.txt", "r") as f:
        sin_data = f.read()
        #print(f"sin_data = {sin_data}")
        #print(type(sin_data))
    # decode sin_data
    sin_data_decoded = json.loads(sin_data, object_hook=multi_type_decoder)
    # print(sin_data_decoded.__dict__)
    run_sin_decoded(sin_data_decoded._features, sin_data_decoded._labels)


if __name__ == "__main__":
    main()


"""
~~~~SAMPLE RUN~~~~
/Users/anne/PycharmProjects/IntSoftwrDsgn_Python/venv/bin/python /Users/anne/PycharmProjects/IntSoftwrDsgn_Python/assignment6.py
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.82                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.78                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.79                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.74                
0.00                
Epoch 0 RMSE =  0.5720971505760925
Epoch 100 RMSE =  0.5097382109624067
Epoch 200 RMSE =  0.5017249792680828
Epoch 300 RMSE =  0.501241477934892
Epoch 400 RMSE =  0.5011386499061544
Epoch 500 RMSE =  0.5010594688406538
Epoch 600 RMSE =  0.500976493007849
Epoch 700 RMSE =  0.5008926046536484
Epoch 800 RMSE =  0.5008051493900906
Epoch 900 RMSE =  0.5007135928698775
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.50                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.50                
0.00                
Epoch 1000 RMSE =  0.5006182853589697
Epoch 1100 RMSE =  0.5005164630300298
Epoch 1200 RMSE =  0.5004076939835461
Epoch 1300 RMSE =  0.5002901625870785
Epoch 1400 RMSE =  0.5001616078082614
Epoch 1500 RMSE =  0.5000190454717023
Epoch 1600 RMSE =  0.4998691003953257
Epoch 1700 RMSE =  0.4996991378775479
Epoch 1800 RMSE =  0.49950970561636443
Epoch 1900 RMSE =  0.49929968652234447
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.51                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.50                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.51                
1.00                
Epoch 2000 RMSE =  0.49906284006705853
Epoch 2100 RMSE =  0.4987979228507901
Epoch 2200 RMSE =  0.49849931264766756
Epoch 2300 RMSE =  0.49816303450127236
Epoch 2400 RMSE =  0.49778299022185596
Epoch 2500 RMSE =  0.49735299781157755
Epoch 2600 RMSE =  0.4968671907853961
Epoch 2700 RMSE =  0.49631750128077284
Epoch 2800 RMSE =  0.49569276052753947
Epoch 2900 RMSE =  0.49499731695453936
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.51                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.52                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.48                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.53                
1.00                
Epoch 3000 RMSE =  0.4942061337765163
Epoch 3100 RMSE =  0.4933283048950797
Epoch 3200 RMSE =  0.4923436632129473
Epoch 3300 RMSE =  0.4912476539514491
Epoch 3400 RMSE =  0.49003391992105283
Epoch 3500 RMSE =  0.48869366356195104
Epoch 3600 RMSE =  0.4872333389004047
Epoch 3700 RMSE =  0.48564088825076346
Epoch 3800 RMSE =  0.48391211115855925
Epoch 3900 RMSE =  0.48205713452226706
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.54                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.54                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.53                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.45                
0.00                
Epoch 4000 RMSE =  0.4800766151304184
Epoch 4100 RMSE =  0.4779647701124419
Epoch 4200 RMSE =  0.47573975686917225
Epoch 4300 RMSE =  0.4733938669167484
Epoch 4400 RMSE =  0.470948656299856
Epoch 4500 RMSE =  0.4683965839869156
Epoch 4600 RMSE =  0.4657636846469887
Epoch 4700 RMSE =  0.4630426913829987
Epoch 4800 RMSE =  0.4602496824126184
Epoch 4900 RMSE =  0.45738642556927955
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.56                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.58                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.41                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.53                
1.00                
Epoch 5000 RMSE =  0.4544722740064656
Epoch 5100 RMSE =  0.45151785672077405
Epoch 5200 RMSE =  0.4485082697699228
Epoch 5300 RMSE =  0.4454766879118734
Epoch 5400 RMSE =  0.4423988640588883
Epoch 5500 RMSE =  0.43929970184047373
Epoch 5600 RMSE =  0.4361843783853526
Epoch 5700 RMSE =  0.43303038719415793
Epoch 5800 RMSE =  0.4298539176631471
Epoch 5900 RMSE =  0.4266447504599117
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.59                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.60                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.50                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.37                
0.00                
Epoch 6000 RMSE =  0.4234023599087907
Epoch 6100 RMSE =  0.42012241563852676
Epoch 6200 RMSE =  0.4167971419436326
Epoch 6300 RMSE =  0.41339540770322575
Epoch 6400 RMSE =  0.409929438637454
Epoch 6500 RMSE =  0.40636661146309316
Epoch 6600 RMSE =  0.40267542919792415
Epoch 6700 RMSE =  0.3988609132107034
Epoch 6800 RMSE =  0.39486577511459464
Epoch 6900 RMSE =  0.3906694479615447
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.45                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.62                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.63                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.34                
0.00                
Epoch 7000 RMSE =  0.3862338426596096
Epoch 7100 RMSE =  0.38152724803632176
Epoch 7200 RMSE =  0.37651061672414654
Epoch 7300 RMSE =  0.3711465378947765
Epoch 7400 RMSE =  0.36540584006501137
Epoch 7500 RMSE =  0.3592837188163402
Epoch 7600 RMSE =  0.3527655424432342
Epoch 7700 RMSE =  0.34586447675418547
Epoch 7800 RMSE =  0.33861600315937557
Epoch 7900 RMSE =  0.33105911823758877
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.36                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.69                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.68                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.29                
0.00                
Epoch 8000 RMSE =  0.32325602766472156
Epoch 8100 RMSE =  0.3152865972581827
Epoch 8200 RMSE =  0.3072338007270426
Epoch 8300 RMSE =  0.2991721930834097
Epoch 8400 RMSE =  0.2911781798000086
Epoch 8500 RMSE =  0.2833205184279419
Epoch 8600 RMSE =  0.27565804911687053
Epoch 8700 RMSE =  0.2682253640953209
Epoch 8800 RMSE =  0.26106122425623235
Epoch 8900 RMSE =  0.2541831678361266
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.76                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.25                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.25                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.76                
1.00                
Epoch 9000 RMSE =  0.2476038861718897
Epoch 9100 RMSE =  0.24132674978958882
Epoch 9200 RMSE =  0.23535106452554355
Epoch 9300 RMSE =  0.22966905221491876
Epoch 9400 RMSE =  0.22427284387075508
Epoch 9500 RMSE =  0.21914797422400517
Epoch 9600 RMSE =  0.2142844196316724
Epoch 9700 RMSE =  0.20966853161116134
Epoch 9800 RMSE =  0.2052841737588681
Epoch 9900 RMSE =  0.20111872191477043
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.81                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.18                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.22                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.81                
0.00                
Epoch 10000 RMSE =  0.19715867464385886
Epoch 10100 RMSE =  0.1933914646303858
Epoch 10200 RMSE =  0.18980511995804156
Epoch 10300 RMSE =  0.18638758167632474
Epoch 10400 RMSE =  0.1831281528993989
Epoch 10500 RMSE =  0.18001779892422165
Epoch 10600 RMSE =  0.17704544243092796
Epoch 10700 RMSE =  0.17420313874637766
Epoch 10800 RMSE =  0.17148273021623112
Epoch 10900 RMSE =  0.1688764989872605
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.14                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.20                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.84                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.84                
0.00                
Epoch 11000 RMSE =  0.16637736345030563
Epoch 11100 RMSE =  0.16397896139658044
Epoch 11200 RMSE =  0.1616756575019176
Epoch 11300 RMSE =  0.1594612043672907
Epoch 11400 RMSE =  0.15733062735190717
Epoch 11500 RMSE =  0.15527907616944542
Epoch 11600 RMSE =  0.15330205784665876
Epoch 11700 RMSE =  0.15139558123192703
Epoch 11800 RMSE =  0.14955582719642527
Epoch 11900 RMSE =  0.14777894774218786
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.86                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.12                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.86                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.18                
0.00                
Epoch 12000 RMSE =  0.14606192117192496
Epoch 12100 RMSE =  0.14440139946712094
Epoch 12200 RMSE =  0.14279454045019196
Epoch 12300 RMSE =  0.141238515251095
Epoch 12400 RMSE =  0.1397311155495205
Epoch 12500 RMSE =  0.13826985911933293
Epoch 12600 RMSE =  0.13685232992935809
Epoch 12700 RMSE =  0.13547651626184923
Epoch 12800 RMSE =  0.13414048680957605
Epoch 12900 RMSE =  0.1328425778301525
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.87                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.88                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.17                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.10                
1.00                
Epoch 13000 RMSE =  0.13158084164123096
Epoch 13100 RMSE =  0.1303538434905163
Epoch 13200 RMSE =  0.12915999739489847
Epoch 13300 RMSE =  0.12799776052536632
Epoch 13400 RMSE =  0.1268659534137442
Epoch 13500 RMSE =  0.12576318826736146
Epoch 13600 RMSE =  0.12468826772331713
Epoch 13700 RMSE =  0.12364020990593907
Epoch 13800 RMSE =  0.12261781480316222
Epoch 13900 RMSE =  0.12162000387861928
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.89                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.16                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.09                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.89                
1.00                
Epoch 14000 RMSE =  0.12064603664113024
Epoch 14100 RMSE =  0.11969483449476909
Epoch 14200 RMSE =  0.11876544475155784
Epoch 14300 RMSE =  0.11785726754340488
Epoch 14400 RMSE =  0.11696947380724756
Epoch 14500 RMSE =  0.11610118827876288
Epoch 14600 RMSE =  0.1152519158218329
Epoch 14700 RMSE =  0.11442076011343794
Epoch 14800 RMSE =  0.11360730255652199
Epoch 14900 RMSE =  0.11281078103607246
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.15                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.08                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.89                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.89                
1.00                
Epoch 15000 RMSE =  0.1120306180118182
Epoch 15100 RMSE =  0.11126643201641881
Epoch 15200 RMSE =  0.11051749509439858
Epoch 15300 RMSE =  0.10978342879282794
Epoch 15400 RMSE =  0.10906364889832483
Epoch 15500 RMSE =  0.10835783671159367
Epoch 15600 RMSE =  0.10766547485094169
Epoch 15700 RMSE =  0.10698615034138384
Epoch 15800 RMSE =  0.10631944069751184
Epoch 15900 RMSE =  0.10566496753957356
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.14                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.90                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.90                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.07                
1.00                
Epoch 16000 RMSE =  0.10502240466106441
Epoch 16100 RMSE =  0.10439133687219075
Epoch 16200 RMSE =  0.1037714631610608
Epoch 16300 RMSE =  0.10316249816091862
Epoch 16400 RMSE =  0.10256403043195618
Epoch 16500 RMSE =  0.10197585622019205
Epoch 16600 RMSE =  0.10139760780801162
Epoch 16700 RMSE =  0.10082905807680895
Epoch 16800 RMSE =  0.1002699651859048
Epoch 16900 RMSE =  0.09971997122961442
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.91                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.13                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.06                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.91                
0.00                
Epoch 17000 RMSE =  0.0991789415648473
Epoch 17100 RMSE =  0.0986465690826015
Epoch 17200 RMSE =  0.09812265227844791
Epoch 17300 RMSE =  0.0976069504561757
Epoch 17400 RMSE =  0.0970992897713081
Epoch 17500 RMSE =  0.09659942374085441
Epoch 17600 RMSE =  0.09610716812116617
Epoch 17700 RMSE =  0.09562232986418602
Epoch 17800 RMSE =  0.09514472485473342
Epoch 17900 RMSE =  0.09467419081344176
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.91                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.06                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.91                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.13                
0.00                
Epoch 18000 RMSE =  0.09421053790164939
Epoch 18100 RMSE =  0.09375361214774555
Epoch 18200 RMSE =  0.0933032356139164
Epoch 18300 RMSE =  0.0928592534729705
Epoch 18400 RMSE =  0.09242155523202751
Epoch 18500 RMSE =  0.09198994860942619
Epoch 18600 RMSE =  0.09156430742747945
Epoch 18700 RMSE =  0.091144490120507
Epoch 18800 RMSE =  0.0907303944617091
Epoch 18900 RMSE =  0.09032185902123459
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.92                
1.00                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.12                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.92                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.06                
1.00                
Epoch 19000 RMSE =  0.08991877131166431
Epoch 19100 RMSE =  0.0895210058846071
Epoch 19200 RMSE =  0.08912843828062376
Epoch 19300 RMSE =  0.08874099707339929
Epoch 19400 RMSE =  0.08835853581033776
Epoch 19500 RMSE =  0.08798096495757965
Epoch 19600 RMSE =  0.08760816676961702
Epoch 19700 RMSE =  0.08724004395950476
Epoch 19800 RMSE =  0.08687650666591891
Epoch 19900 RMSE =  0.08651744605597857
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.12                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.92                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.92                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.05                
1.00                
Epoch 20000 RMSE =  0.08616280889786523
Final Training Epoch RMSE =  0.08616280889786523
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.12                
0.00                
Input Values        Expected Values     Predicted Values    
1.00                1.00                0.92                
0.00                
Input Values        Expected Values     Predicted Values    
0.00                1.00                0.92                
1.00                
Input Values        Expected Values     Predicted Values    
1.00                0.00                0.05                
1.00                
Test Epoch RMSE =  0.08614001599098589

Input Values        Expected Values     Predicted Values    
0.73                0.67                0.85                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.83                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.82                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.88                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.86                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.84                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.84                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.83                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.87                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.83                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.85                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.87                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.81                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.85                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.84                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.86                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.82                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.86                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.87                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.84                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.85                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.84                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.86                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.85                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.86                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.87                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.81                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.81                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.86                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.83                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.81                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.82                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.81                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.83                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.84                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.81                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.84                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.86                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.82                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.85                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.80                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.86                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.81                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.81                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.85                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.80                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.80                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.80                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.81                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.82                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.80                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.85                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.84                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.82                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.80                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.85                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.82                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.82                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.84                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.83                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.83                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.84                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.82                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.85                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.82                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.83                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.83                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.85                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.82                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.81                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.83                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.85                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.84                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.81                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.83                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.81                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.80                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.83                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.81                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.82                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.81                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.84                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.84                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.82                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.84                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.80                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.84                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.84                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.84                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.82                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.84                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.84                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.79                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.80                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.83                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.81                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.81                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.79                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.83                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.80                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.79                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.82                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.80                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.79                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.84                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.81                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.78                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.81                
Epoch 0 RMSE =  0.37336525682963023
Epoch 100 RMSE =  0.12010126976363586
Epoch 200 RMSE =  0.04934751471100073
Epoch 300 RMSE =  0.037335483148579554
Epoch 400 RMSE =  0.03445119504099268
Epoch 500 RMSE =  0.03336460914036069
Epoch 600 RMSE =  0.03271420059123109
Epoch 700 RMSE =  0.0322163448994873
Epoch 800 RMSE =  0.03179954889175994
Epoch 900 RMSE =  0.031442249585880315
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.95                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.57                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.12                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.53                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.86                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.76                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.94                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.40                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.93                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.94                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.87                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.46                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.94                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.12                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.89                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.55                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.72                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.64                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.11                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.09                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.50                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.08                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.93                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.39                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.52                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.36                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.94                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.78                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.95                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.19                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.58                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.32                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.92                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.93                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.17                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.35                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.22                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.95                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.61                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.15                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.91                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.18                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.49                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.92                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.09                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.71                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.34                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.09                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.75                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.92                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.08                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.48                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.91                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.89                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.10                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.89                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.38                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.17                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.31                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.94                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.94                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.87                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.56                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.95                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.68                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.91                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.91                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.11                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.47                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.62                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.10                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Epoch 1000 RMSE =  0.03113257255529826
Epoch 1100 RMSE =  0.030862032700358356
Epoch 1200 RMSE =  0.030622200851326175
Epoch 1300 RMSE =  0.030408654594085725
Epoch 1400 RMSE =  0.03021916598219296
Epoch 1500 RMSE =  0.030047099418413534
Epoch 1600 RMSE =  0.029890418665185707
Epoch 1700 RMSE =  0.029752021407921457
Epoch 1800 RMSE =  0.029623957821001128
Epoch 1900 RMSE =  0.02950590138145231
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.56                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.19                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.57                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.55                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.52                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.95                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.89                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.31                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.61                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.50                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.38                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.39                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.62                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.08                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.93                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.87                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.15                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.92                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.10                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.76                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.36                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.40                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.17                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.09                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.95                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.71                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.91                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.35                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.09                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.94                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.10                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.22                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.11                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.94                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.89                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.91                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.53                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.12                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.34                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.58                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.95                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.32                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.78                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.64                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Epoch 2000 RMSE =  0.029398027326971696
Epoch 2100 RMSE =  0.02929870870005119
Epoch 2200 RMSE =  0.029205382255601972
Epoch 2300 RMSE =  0.029120889616801985
Epoch 2400 RMSE =  0.02904059045052476
Epoch 2500 RMSE =  0.02896719009972379
Epoch 2600 RMSE =  0.02889547564558755
Epoch 2700 RMSE =  0.028832698802968827
Epoch 2800 RMSE =  0.028771568295155942
Epoch 2900 RMSE =  0.02871416501976125
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.40                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.08                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.10                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.58                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.22                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.30                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.87                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.57                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.62                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.38                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.53                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.94                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.09                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.89                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.17                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.76                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.91                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.19                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.55                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.39                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.15                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.56                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.61                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.95                
Epoch 3000 RMSE =  0.028660066358645445
Epoch 3100 RMSE =  0.02860929544986244
Epoch 3200 RMSE =  0.02856072039070926
Epoch 3300 RMSE =  0.028513651979841372
Epoch 3400 RMSE =  0.028472151388146037
Epoch 3500 RMSE =  0.02843113174016798
Epoch 3600 RMSE =  0.028390268674527798
Epoch 3700 RMSE =  0.028354620588670634
Epoch 3800 RMSE =  0.028318977179653385
Epoch 3900 RMSE =  0.028286213386985032
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.10                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.31                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.89                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.15                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.39                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.09                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.58                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.19                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.40                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.91                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.17                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.08                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Epoch 4000 RMSE =  0.028253372785099876
Epoch 4100 RMSE =  0.028223483057380008
Epoch 4200 RMSE =  0.028193904597256486
Epoch 4300 RMSE =  0.02816359439963977
Epoch 4400 RMSE =  0.02813711305253207
Epoch 4500 RMSE =  0.02810786393152965
Epoch 4600 RMSE =  0.02808817691830422
Epoch 4700 RMSE =  0.028064535756234528
Epoch 4800 RMSE =  0.028042307238670947
Epoch 4900 RMSE =  0.02801884356125236
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.40                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.69                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.20                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.38                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.30                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.17                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.10                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.57                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.86                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.16                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.08                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.10                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Epoch 5000 RMSE =  0.027998416417963332
Epoch 5100 RMSE =  0.027978334781569174
Epoch 5200 RMSE =  0.027958082486512405
Epoch 5300 RMSE =  0.027939432280117384
Epoch 5400 RMSE =  0.02792111388943252
Epoch 5500 RMSE =  0.02790370988626766
Epoch 5600 RMSE =  0.027886070704506834
Epoch 5700 RMSE =  0.02787057557898991
Epoch 5800 RMSE =  0.027853850245909677
Epoch 5900 RMSE =  0.02783886265634342
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.42                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.08                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.10                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.31                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.18                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.39                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.82                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.38                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.69                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.41                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.10                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.16                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.87                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.57                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.20                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Epoch 6000 RMSE =  0.02782470107636389
Epoch 6100 RMSE =  0.02780537013840395
Epoch 6200 RMSE =  0.027796075641197772
Epoch 6300 RMSE =  0.027782148300190893
Epoch 6400 RMSE =  0.027769127354054693
Epoch 6500 RMSE =  0.02775658837465077
Epoch 6600 RMSE =  0.02774355775561738
Epoch 6700 RMSE =  0.027732093867187
Epoch 6800 RMSE =  0.02771953586301964
Epoch 6900 RMSE =  0.027709395033617506
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.14                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.69                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.40                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.87                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.30                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.09                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.10                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.16                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.38                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.39                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.20                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.57                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.11                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.18                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.41                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Epoch 7000 RMSE =  0.02769701171455263
Epoch 7100 RMSE =  0.027686426168898232
Epoch 7200 RMSE =  0.02767676082622703
Epoch 7300 RMSE =  0.027666318896972544
Epoch 7400 RMSE =  0.027657199019723123
Epoch 7500 RMSE =  0.02764627789344851
Epoch 7600 RMSE =  0.0276375308987298
Epoch 7700 RMSE =  0.027629235849307026
Epoch 7800 RMSE =  0.027619253042712916
Epoch 7900 RMSE =  0.027611673353474305
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.94                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.16                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.39                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.41                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.40                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.10                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.18                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.69                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.57                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.30                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.82                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.11                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.87                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.15                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.38                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.95                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.20                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.09                
Epoch 8000 RMSE =  0.027603564844476042
Epoch 8100 RMSE =  0.0275953886849379
Epoch 8200 RMSE =  0.027587088620402607
Epoch 8300 RMSE =  0.0275796955336379
Epoch 8400 RMSE =  0.02757210119426963
Epoch 8500 RMSE =  0.027564099322859487
Epoch 8600 RMSE =  0.027556432244345164
Epoch 8700 RMSE =  0.02754969091698677
Epoch 8800 RMSE =  0.027542927196690725
Epoch 8900 RMSE =  0.02753599901698767
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.38                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.11                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.87                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.39                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.69                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.95                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.57                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.13                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.20                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.10                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.15                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.16                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.11                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.18                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.40                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.81                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.23                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.93                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.09                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.42                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.95                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.94                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.41                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.30                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Epoch 9000 RMSE =  0.02752818636584232
Epoch 9100 RMSE =  0.02752316764383648
Epoch 9200 RMSE =  0.02751705746494086
Epoch 9300 RMSE =  0.02751001990755192
Epoch 9400 RMSE =  0.027504161171903786
Epoch 9500 RMSE =  0.027498789228808673
Epoch 9600 RMSE =  0.027490648953788315
Epoch 9700 RMSE =  0.027487112836176806
Epoch 9800 RMSE =  0.027480017816697153
Epoch 9900 RMSE =  0.02747602302318196
Input Values        Expected Values     Predicted Values    
1.49                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.35                0.34                0.31                
Input Values        Expected Values     Predicted Values    
0.02                0.02                0.09                
Input Values        Expected Values     Predicted Values    
0.22                0.22                0.20                
Input Values        Expected Values     Predicted Values    
1.50                1.00                0.96                
Input Values        Expected Values     Predicted Values    
0.27                0.27                0.24                
Input Values        Expected Values     Predicted Values    
0.33                0.32                0.30                
Input Values        Expected Values     Predicted Values    
0.55                0.52                0.52                
Input Values        Expected Values     Predicted Values    
1.22                0.94                0.93                
Input Values        Expected Values     Predicted Values    
0.07                0.07                0.12                
Input Values        Expected Values     Predicted Values    
1.39                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.11                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.04                0.04                0.10                
Input Values        Expected Values     Predicted Values    
1.27                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.56                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.42                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.15                0.15                0.16                
Input Values        Expected Values     Predicted Values    
0.30                0.30                0.27                
Input Values        Expected Values     Predicted Values    
0.42                0.41                0.38                
Input Values        Expected Values     Predicted Values    
0.43                0.42                0.39                
Input Values        Expected Values     Predicted Values    
1.43                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.84                0.74                0.77                
Input Values        Expected Values     Predicted Values    
1.51                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.69                0.64                0.66                
Input Values        Expected Values     Predicted Values    
0.98                0.83                0.85                
Input Values        Expected Values     Predicted Values    
1.03                0.86                0.87                
Input Values        Expected Values     Predicted Values    
0.49                0.47                0.46                
Input Values        Expected Values     Predicted Values    
0.70                0.64                0.67                
Input Values        Expected Values     Predicted Values    
1.46                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.94                0.81                0.83                
Input Values        Expected Values     Predicted Values    
0.26                0.26                0.24                
Input Values        Expected Values     Predicted Values    
1.33                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.63                0.59                0.60                
Input Values        Expected Values     Predicted Values    
0.34                0.33                0.31                
Input Values        Expected Values     Predicted Values    
0.08                0.08                0.12                
Input Values        Expected Values     Predicted Values    
0.91                0.79                0.82                
Input Values        Expected Values     Predicted Values    
0.88                0.77                0.80                
Input Values        Expected Values     Predicted Values    
0.76                0.69                0.72                
Input Values        Expected Values     Predicted Values    
0.41                0.40                0.37                
Input Values        Expected Values     Predicted Values    
0.25                0.25                0.23                
Input Values        Expected Values     Predicted Values    
0.65                0.61                0.62                
Input Values        Expected Values     Predicted Values    
0.92                0.80                0.82                
Input Values        Expected Values     Predicted Values    
1.44                0.99                0.96                
Input Values        Expected Values     Predicted Values    
1.32                0.97                0.94                
Input Values        Expected Values     Predicted Values    
0.32                0.31                0.29                
Input Values        Expected Values     Predicted Values    
0.59                0.56                0.56                
Input Values        Expected Values     Predicted Values    
1.06                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.23                0.23                0.21                
Input Values        Expected Values     Predicted Values    
0.37                0.36                0.33                
Input Values        Expected Values     Predicted Values    
1.12                0.90                0.90                
Input Values        Expected Values     Predicted Values    
1.30                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.04                0.86                0.87                
Input Values        Expected Values     Predicted Values    
1.55                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.13                0.13                0.15                
Input Values        Expected Values     Predicted Values    
1.37                0.98                0.95                
Input Values        Expected Values     Predicted Values    
0.21                0.21                0.20                
Input Values        Expected Values     Predicted Values    
0.46                0.44                0.43                
Input Values        Expected Values     Predicted Values    
1.57                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.10                0.10                0.13                
Input Values        Expected Values     Predicted Values    
0.81                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.80                0.72                0.75                
Input Values        Expected Values     Predicted Values    
0.00                0.00                0.09                
Input Values        Expected Values     Predicted Values    
1.25                0.95                0.93                
Input Values        Expected Values     Predicted Values    
1.52                1.00                0.97                
Input Values        Expected Values     Predicted Values    
0.18                0.18                0.18                
Input Values        Expected Values     Predicted Values    
0.19                0.19                0.18                
Input Values        Expected Values     Predicted Values    
0.09                0.09                0.12                
Input Values        Expected Values     Predicted Values    
0.60                0.56                0.57                
Input Values        Expected Values     Predicted Values    
1.53                1.00                0.97                
Input Values        Expected Values     Predicted Values    
1.41                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.79                0.71                0.74                
Input Values        Expected Values     Predicted Values    
0.28                0.28                0.25                
Input Values        Expected Values     Predicted Values    
0.74                0.67                0.70                
Input Values        Expected Values     Predicted Values    
0.11                0.11                0.14                
Input Values        Expected Values     Predicted Values    
1.19                0.93                0.92                
Input Values        Expected Values     Predicted Values    
1.47                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.48                0.46                0.45                
Input Values        Expected Values     Predicted Values    
0.87                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.58                0.55                0.55                
Input Values        Expected Values     Predicted Values    
1.02                0.85                0.87                
Input Values        Expected Values     Predicted Values    
0.66                0.61                0.63                
Input Values        Expected Values     Predicted Values    
1.21                0.94                0.92                
Input Values        Expected Values     Predicted Values    
0.96                0.82                0.84                
Input Values        Expected Values     Predicted Values    
0.64                0.60                0.61                
Input Values        Expected Values     Predicted Values    
0.50                0.48                0.47                
Input Values        Expected Values     Predicted Values    
0.20                0.20                0.19                
Input Values        Expected Values     Predicted Values    
0.57                0.54                0.54                
Input Values        Expected Values     Predicted Values    
0.47                0.45                0.44                
Input Values        Expected Values     Predicted Values    
0.06                0.06                0.11                
Input Values        Expected Values     Predicted Values    
0.75                0.68                0.71                
Input Values        Expected Values     Predicted Values    
0.44                0.43                0.40                
Input Values        Expected Values     Predicted Values    
1.45                0.99                0.96                
Input Values        Expected Values     Predicted Values    
0.73                0.67                0.69                
Input Values        Expected Values     Predicted Values    
1.08                0.88                0.89                
Input Values        Expected Values     Predicted Values    
0.38                0.37                0.34                
Input Values        Expected Values     Predicted Values    
1.34                0.97                0.95                
Input Values        Expected Values     Predicted Values    
0.39                0.38                0.35                
Input Values        Expected Values     Predicted Values    
1.10                0.89                0.89                
Input Values        Expected Values     Predicted Values    
1.13                0.90                0.90                
Input Values        Expected Values     Predicted Values    
0.54                0.51                0.51                
Input Values        Expected Values     Predicted Values    
0.52                0.50                0.49                
Input Values        Expected Values     Predicted Values    
0.45                0.43                0.41                
Input Values        Expected Values     Predicted Values    
1.28                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.07                0.88                0.88                
Input Values        Expected Values     Predicted Values    
1.20                0.93                0.92                
Input Values        Expected Values     Predicted Values    
0.51                0.49                0.48                
Input Values        Expected Values     Predicted Values    
0.05                0.05                0.11                
Input Values        Expected Values     Predicted Values    
1.05                0.87                0.88                
Input Values        Expected Values     Predicted Values    
0.86                0.76                0.79                
Input Values        Expected Values     Predicted Values    
0.03                0.03                0.10                
Input Values        Expected Values     Predicted Values    
0.31                0.31                0.28                
Epoch 10000 RMSE =  0.027470383581592032
Final Training Epoch RMSE =  0.027470383581592032
Input Values        Expected Values     Predicted Values    
0.01                0.01                0.09                
Input Values        Expected Values     Predicted Values    
0.12                0.12                0.14                
Input Values        Expected Values     Predicted Values    
0.14                0.14                0.15                
Input Values        Expected Values     Predicted Values    
0.16                0.16                0.16                
Input Values        Expected Values     Predicted Values    
0.17                0.17                0.17                
Input Values        Expected Values     Predicted Values    
0.24                0.24                0.22                
Input Values        Expected Values     Predicted Values    
0.29                0.29                0.26                
Input Values        Expected Values     Predicted Values    
0.36                0.35                0.32                
Input Values        Expected Values     Predicted Values    
0.40                0.39                0.36                
Input Values        Expected Values     Predicted Values    
0.53                0.51                0.50                
Input Values        Expected Values     Predicted Values    
0.56                0.53                0.53                
Input Values        Expected Values     Predicted Values    
0.61                0.57                0.58                
Input Values        Expected Values     Predicted Values    
0.62                0.58                0.59                
Input Values        Expected Values     Predicted Values    
0.67                0.62                0.64                
Input Values        Expected Values     Predicted Values    
0.68                0.63                0.65                
Input Values        Expected Values     Predicted Values    
0.71                0.65                0.67                
Input Values        Expected Values     Predicted Values    
0.72                0.66                0.68                
Input Values        Expected Values     Predicted Values    
0.77                0.70                0.72                
Input Values        Expected Values     Predicted Values    
0.78                0.70                0.73                
Input Values        Expected Values     Predicted Values    
0.82                0.73                0.76                
Input Values        Expected Values     Predicted Values    
0.83                0.74                0.77                
Input Values        Expected Values     Predicted Values    
0.85                0.75                0.78                
Input Values        Expected Values     Predicted Values    
0.89                0.78                0.80                
Input Values        Expected Values     Predicted Values    
0.90                0.78                0.81                
Input Values        Expected Values     Predicted Values    
0.93                0.80                0.83                
Input Values        Expected Values     Predicted Values    
0.95                0.81                0.84                
Input Values        Expected Values     Predicted Values    
0.97                0.82                0.85                
Input Values        Expected Values     Predicted Values    
0.99                0.84                0.85                
Input Values        Expected Values     Predicted Values    
1.00                0.84                0.86                
Input Values        Expected Values     Predicted Values    
1.01                0.85                0.86                
Input Values        Expected Values     Predicted Values    
1.09                0.89                0.89                
Input Values        Expected Values     Predicted Values    
1.14                0.91                0.91                
Input Values        Expected Values     Predicted Values    
1.15                0.91                0.91                
Input Values        Expected Values     Predicted Values    
1.16                0.92                0.91                
Input Values        Expected Values     Predicted Values    
1.17                0.92                0.91                
Input Values        Expected Values     Predicted Values    
1.18                0.92                0.92                
Input Values        Expected Values     Predicted Values    
1.23                0.94                0.93                
Input Values        Expected Values     Predicted Values    
1.24                0.95                0.93                
Input Values        Expected Values     Predicted Values    
1.26                0.95                0.93                
Input Values        Expected Values     Predicted Values    
1.29                0.96                0.94                
Input Values        Expected Values     Predicted Values    
1.31                0.97                0.94                
Input Values        Expected Values     Predicted Values    
1.35                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.36                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.38                0.98                0.95                
Input Values        Expected Values     Predicted Values    
1.40                0.99                0.95                
Input Values        Expected Values     Predicted Values    
1.48                1.00                0.96                
Input Values        Expected Values     Predicted Values    
1.54                1.00                0.97                
Test Epoch RMSE =  0.02404266573968525

Process finished with exit code 0

"""