import math
import numpy as np


class VanEmdeBoas:
    """
    Attempt to implement a Van Emde Boas queue.
    """

    def __init__(self, size, key=lambda x: x):
        self.M = math.pow(2, size / 2)
        self.__key = key
        self.min = self.M
        self.max = -1
        self.children = []
        self.aux = None

        # Base case
        if size > 2:
            no_clusters = math.ceil(math.sqrt(size))

            # Assigning VEB(sqrt(u)) to summary
            self.aux = VanEmdeBoas(no_clusters, key=self.__key)

            self.children = [VanEmdeBoas(no_clusters, key=self.__key) for _ in range(no_clusters)]
            # Assigning VEB(sqrt(u)) to all of its clusters

    def high(self, x):
        return math.floor(x / math.ceil(math.sqrt(self.M)))

    def low(self, x):
        return x % math.ceil(math.sqrt(self.M))

    def index(self, x, y):
        return x * math.ceil(math.sqrt(self.M)) + y

    # def key(self, x):
    #     if x == -1 or x == self.M or isinstance(x, float) or isinstance(x, int):
    #         return x
    #     return self.__key(x)

    def successor(self, x):
        if x < self.min:
            return self.min
        elif x >= self.max:
            return self.M

        i = self.high(x)
        j = self.low(x)
        print(i, len(self.children))
        if len(self.children) > 0 and j < self.children[i].max:
            return math.floor(x - j + self.children[i].successor(j))
        # otherwise find next subtree for successor
        return math.floor(x - j + self.children[self.aux.successor(i)].min)

    def insert(self, x):
        if self.min > self.max:  # tree is empty
            self.min = self.max = x
            return
        if x < self.min:
            x, self.min = self.min, x
        if x > self.max:
            self.max = x

        if self.M == 2:
            self.max = x
            return

        i = self.high(x)
        j = self.low(x)

        self.children[i].insert(j)
        if self.children[i].min == self.children[i].max:
            self.aux.insert(i)

    def delete(self, x):
        if self.min > self.max:
            return

        if self.min == self.max:
            self.min = self.M
            self.max = -1
            return

        if self.M == 2:  # if last subtree
            if x == 1:
                if self.min == 1:
                    self.min = self.M
                    self.max = -1
                elif self.min == 0:
                    self.max = 0
            else:
                # i.e. x == 0
                if self.max == 0:
                    self.min = self.M
                    self.max = -1
                elif self.max == 1:
                    self.min = 1
            return

        # if x == self.min:
        #     i = math.floor(self.key(self.aux.min))
        #     self.min = i * math.sqrt(self.M) + self.key(self.children[i].min)
        #     return

        if x == self.min:
            # self.min = self.successor(x)
            hi = self.aux.min * math.sqrt(self.M)
            j = math.ceil(self.aux.min)
            self.min = x = hi + self.children[j].min

        i = self.high(x)
        lo = self.low(x)

        self.children[i].delete(lo)
        if self.children[i].min > self.children[i].max:
            self.aux.delete(i)

        if x == self.max:
            if self.aux.min > self.aux.max:
                self.max = self.min
            else:
                # i = math.floor(self.key(self.aux.min))
                # self.min = i * math.sqrt(self.M) + self.key(self.children[i].min)
                hi = self.aux.max * math.sqrt(self.M)
                j = math.floor(self.aux.max)
                self.max = hi + self.children[j].max


class BucketQueue:
    """
    Attempt to implement a bucket queue.
    """

    def __init__(self):
        self.__nbBuckets = 0
        self.buckets = None
        self.key = None
        self.__M = None

    def __iter__(self):
        for i in range(self.__nbBuckets):
            yield self.buckets[i]

    def __len__(self):
        return len([item for bucket in self.buckets for item in bucket])

    def __str__(self):
        l = []
        for i in range(self.__nbBuckets):
            l += self.buckets[i]
        return str(l)

    def __getIndex(self, k):
        """
        Return the index of the bucket that should contain the item.

        :param k: The item.
        :return:
        """
        i = np.ceil(self.__nbBuckets * self.key(k) / self.__M) - 1
        if i < 0 or np.isnan(i):
            i = 0
        return int(i)

    def insert(self, k):
        """
        Insert the item in the corresponding bucket.

        :param k:
        :return:
        """
        if self.__M is None or self.key(k) > self.__M:
            # raise ValueError("Value to insert cannot be superior to ")
            l = [item for bucket in self.buckets for item in bucket]
            l.append(k)
            self.__bucketSort(l)
        else:
            i = self.__getIndex(k)
            self.buckets[i].append(k)
            self.buckets[i] = sorted(self.buckets[i])

    def delete_min(self):
        """
        Remove the minimal value of all the buckets joined.

        :return: The value.
        """
        for bucket in self.buckets:
            if len(bucket) > 0:
                return_value = bucket[0]
                bucket.pop(0)
                return return_value

    def change_key(self, old, new):
        """
        Change the old item to the new one by removing the old one and inserting the new one.

        :param old: The old item
        :param new: The new item
        :return:
        """
        if not np.isinf(self.key(old)):
            i = self.__getIndex(old)
            if old in self.buckets[i]:
                self.buckets[i].remove(old)
        self.insert(new)

    def build_heap(self, nbBuckets: int, aList: list, max_cost=0, key=lambda x: x):
        """
        The constructor for our bucket queue.

        :param nbBuckets: Number of buckets to use.
        :param aList: The starting list.
        :param max_cost: The maximum cost that an element in the queue can take.
        :param key: Define what to mesure for the values.
        :return:
        """
        self.__M = max_cost
        self.__nbBuckets = nbBuckets
        self.key = key
        self.buckets = [[] for _ in range(self.__nbBuckets)]
        if len(aList) > 0:
            self.__bucketSort(aList)

    def __bucketSort(self, array):
        """
        Bucket sort for the initialization.

        :param array: Array to sort
        :return:
        """
        self.__M = self.key(max(array, key=self.key))
        if self.__nbBuckets <= 0:
            self.__nbBuckets = self.__M // len(array)

        for i in range(len(array)):
            self.buckets[self.__getIndex(array[i])].append(array[i])

        for i in range(self.__nbBuckets):
            self.buckets[i] = sorted(self.buckets[i])


class BinaryHeap:
    """
    In order for our heap to work efficiently, we will take advantage of the logarithmic nature of the binary tree to
    represent our heap. In order to guarantee logarithmic performance, we must keep our tree balanced. A balanced
    binary tree has roughly the same number of nodes in the left and right subtrees of the root. In our heap
    implementation we keep the tree balanced by creating a complete binary tree. A complete binary tree is a tree in
    which each level has all of its nodes. The exception to this is the bottom level of the tree, which we fill in
    from left to right.
    Another interesting property of a complete tree is that we can represent it using a single
    list. We do not need to use nodes and references or even lists of lists. Because the tree is complete,
    the left child of a parent (at position p) is the node that is found in position 2p in the list. Similarly,
    the right child of the parent is at position 2p+1 in the list. To find the parent of any node in the
    tree, we can simply use integer division (like normal mathematical division except we discard the remainder).
    Given that a node is at position nnn in the list, the parent is at position n/2.
    The method that we will use to store items in a heap relies on maintaining the heap order property. The heap order
    property is as follows: In a heap, for every node x with parent p, the key in p is smaller than or equal to the
    key in x.
    """

    def __init__(self):
        self.items = []
        self.key = None

    def __len__(self):
        return len(self.items) - 1

    def __iter__(self):
        for item in self.items[1:]:
            yield item

    def __str__(self):
        return str(self.items)

    def isEmpty(self):
        return len(self.items[1:]) == 0

    def percolate_up(self):
        """
        When we percolate an item up, we are restoring the heap property between the newly added item and
        the parent. We are also preserving the heap property for any siblings. Of course, if the newly added item is
        very small, we may still need to swap it up another level. In fact, we may need to keep swapping until we get
        to the top of the tree.
        Notice that we can compute the parent of any node by using simple integer division. The parent of the current
        node can be computed by dividing the index of the current node by 2.

        :return:
        """
        i = len(self)
        while i // 2 > 0:
            if self.key(self.items[i]) < self.key(self.items[i // 2]):
                self.items[i // 2], self.items[i] = self.items[i], self.items[i // 2]
            i = i // 2

    def insert(self, k):
        """
        The easiest, and most efficient, way to add an item to a list is to simply append the item to the end of the
        list. The good news about appending is that it guarantees that we will maintain the complete tree property.
        The bad news about appending is that we will very likely violate the heap structure property. However,
        it is possible to write a method that will allow us to regain the heap structure property by comparing the
        newly added item with its parent. If the newly added item is less than its parent, then we can swap the item
        with its parent.
        Most of the work in the insert method is really done by percolate_up. Once a new item is appended to the tree,
        percolate_up takes over and positions the new item properly.

        :param k: Element to append to the tree
        :return:
        """
        self.items.append(k)
        self.percolate_up()

    def percolate_down(self, i):
        """
        In order to maintain the heap order property, all we need to do is swap the root with its smallest child less
        than the root. After the initial swap, we may repeat the swapping process with a node and its children until
        the node is swapped into a position on the tree where it is already less than both children.

        :param i: The node to percolate down the tree
        :return:
        """
        while i * 2 <= len(self):
            mc = self.min_child(i)
            if self.key(self.items[i]) > self.key(self.items[mc]):
                self.items[i], self.items[mc] = self.items[mc], self.items[i]
            i = mc

    def min_child(self, i):
        """
        Finds the minimum child to swap the parent with.

        :param i: The node to percolate down the tree
        :return:
        """
        if i * 2 + 1 > len(self):
            return i * 2

        if self.key(self.items[i * 2]) < self.key(self.items[i * 2 + 1]):
            return i * 2

        return i * 2 + 1

    def delete_min(self):
        """
        Since the heap property requires that the root of the tree be the smallest item in the tree, finding the
        minimum item is easy. The hard part of delete_min is restoring full compliance with the heap structure and
        heap order properties after the root has been removed. We can restore our heap in two steps.
        First, we will restore the root item by taking the last item in the list and moving it to the root position.
        Moving the last item maintains our heap structure property. However, we have probably destroyed the heap order
        property of our binary heap.
        Second, we will restore the heap order property by pushing the new root node down the tree to its proper
        position.

        :return:
        """
        return_value = self.items[1]
        self.items[1] = self.items[len(self)]
        self.items.pop()
        self.percolate_down(1)
        return return_value

    def build_heap(self, aList: list, key=lambda x: x):
        """
        We need to build the list. The first method you might think of may be like the following. Given a list of keys,
        you could easily build a heap by inserting each key one at a time. Since you are starting with a list of one
        item, the list is sorted and you could use binary search to find the right position to insert the next key at a
        cost of approximately O(log n) operations. However, remember that inserting an item in the middle of the list
        may require O(n) operations to shift the rest of the list over to make room for the new key. Therefore, to
        insert n keys into the heap would require a total of O(nlog n) operations. However, if we start with an entire
        list then we can build the whole heap in O(n) operations.

        :param aList: (list) The list to start with.
        :param key: The lambda expression defining what is the key in the list. Default is the item itself.
        :return:
        """
        self.key = key
        i = len(aList) // 2
        self.items = [0] + aList
        while i > 0:  # n
            self.percolate_down(i)
            i = i - 1
