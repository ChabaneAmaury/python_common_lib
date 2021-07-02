def bucketSort(array, nbBuckets=0, key=lambda x: x):
    M = max(array, key=key)
    if nbBuckets <= 0:
        nbBuckets = int(M / len(array))
    buckets = [[] for _ in range(nbBuckets + 1)]

    for i in range(len(array)):
        buckets[int(nbBuckets * key(array[i]) / M)].append(array[i])

    for i in range(nbBuckets):
        buckets[i] = sorted(buckets[i])

    return [item for bucket in buckets for item in bucket]


def heapSort(array, key=lambda x: x):
    """
    We need to build the list. The first method you might think of may be like the following. Given a list of keys,
    you could easily build a heap by inserting each key one at a time. Since you are starting with a list of one
    item, the list is sorted and you could use binary search to find the right position to insert the next key at a
    cost of approximately O(log n) operations. However, remember that inserting an item in the middle of the list
    may require O(n) operations to shift the rest of the list over to make room for the new key. Therefore, to
    insert n keys into the heap would require a total of O(nlog n) operations. However, if we start with an entire
    list then we can build the whole heap in O(n) operations.

    :param array: (list) The list to start with.
    :param key: The lambda expression defining what is the key in the list. Default is the item itself.
    :return: items: The sorted list.
    """
    def min_child(i):
        nonlocal items
        """
        Finds the minimum child to swap the parent with.

        :param i: The node to percolate down the tree
        :return:
        """
        if i * 2 + 1 > len(items) - 1:
            return i * 2

        if key(items[i * 2]) < key(items[i * 2 + 1]):
            return i * 2

        return i * 2 + 1

    def percolate_down(i):
        """
        In order to maintain the heap order property, all we need to do is swap the root with its smallest child less
        than the root. After the initial swap, we may repeat the swapping process with a node and its children until
        the node is swapped into a position on the tree where it is already less than both children.

        :param i: The node to percolate down the tree
        :return:
        """
        nonlocal items
        while i * 2 <= len(items) - 1:
            mc = min_child(i)
            if key(items[i]) > key(items[mc]):
                items[i], items[mc] = items[mc], items[i]
            i = mc

    i = len(array) // 2
    items = [0] + array
    while i > 0:
        percolate_down(i)
        i = i - 1

    try:
        items.pop(0)
    except IndexError:
        pass
    return items
