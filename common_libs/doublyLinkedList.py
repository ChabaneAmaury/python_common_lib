class Node:
    def __init__(self, data):
        self.element = data
        self.next = None
        self.previous = None


class doublyLinkedList:
    def __init__(self):
        self.start_node = None
        self.__len = 0

    def __len__(self):
        return self.__len

    def insertEmptyList(self, data):
        """
        Insert a data inside an empty list. Does not insert if the list is not empty.

        :param data: (Any) The data to insert
        :return:
        """
        if self.start_node is None:
            self.start_node = Node(data)
            self.__len += 1
        else:
            print("List is not empty")

    def insertStart(self, data):
        """
        Insert the data at the start position.

        :param data: (Any) The data to insert
        :return:
        """
        if self.start_node is None:
            self.start_node = Node(data)
            self.__len += 1
        else:
            newNode = Node(data)
            newNode.next = self.start_node
            self.start_node.previous = newNode
            self.start_node = newNode
            self.__len += 1

    def insertEnd(self, data):
        """
        Insert the data at the end of the list.

        :param data: (Any) The data to insert
        :return:
        """
        if self.start_node is None:
            self.start_node = Node(data)
            self.__len += 1
        else:
            n = self.start_node
            while n.next is not None:
                n = n.next
            newNode = Node(data)
            n.next, newNode.previous = newNode, n
            self.__len += 1

    def __findNodeInList(self, x, data):
        if self.start_node is None:
            print("List is empty")
            return None, None
        else:
            n = self.start_node
            while n is not None:
                if n.element == x:
                    break
                n = n.next
            if n is None:
                print("Element {0} does not exist in the list".format(x))
                return None, None
            else:
                newNode = Node(data)
                return n, newNode

    def insertAfter(self, x, data):
        """
        Insert the data after the x element.

        :param x: (Any) The data to look for
        :param data: (Any) The data to insert
        :return:
        """

        n, newNode = self.__findNodeInList(x, data)
        if n is None or newNode is not None:
            newNode.previous, newNode.next = n, n.next
            if n.next is not None:
                n.next.previous = newNode
            n.next = newNode
            self.__len += 1

    def insertBefore(self, x, data):
        """
        Insert the data before the x element.

        :param x: (Any) The data to look for
        :param data: (Any) The data to insert
        :return:
        """

        n, newNode = self.__findNodeInList(x, data)
        if n is None or newNode is not None:
            newNode.next, newNode.previous = n, n.previous
            if n.previous is not None:
                n.previous.next = newNode
            n.previous = newNode
            self.__len += 1

    def insertLinkedListAfter(self, linkedList):  # n
        """
        Insert the given doubly linked list after the element based on the first element of the given list.
        Note: The first element is not duplicated.

        :param linkedList: (doublyLinkedList) The doubly linked list to insert
        :return:
        """

        n, newNode = self.__findNodeInList(linkedList.start_node.element, None)
        newNode = linkedList.start_node.next
        if n is not None and newNode is not None:
            n.next, linkedList.start_node.next.previous, n.next.previous, linkedList.getLastElement().next = linkedList.start_node.next, n, linkedList.getLastElement(), n.next
            self.__len += len(linkedList)

    def insertLinkedListBefore(self, linkedList):  # n
        """
        Insert the given doubly linked list before the element based on the last element of the given list.
        Note: The first element is not duplicated.

        :param linkedList: (doublyLinkedList) The doubly linked list to insert
        :return:
        """

        n, newNode = self.__findNodeInList(linkedList.getLastElement().element, None)
        newNode = linkedList.getLastElement().previous
        if n is not None and newNode is not None:
            n.previous.next, linkedList.start_node.previous, n.previous, linkedList.getLastElement().previous = linkedList.start_node, n.previous, linkedList.getLastElement().previous, n
            self.__len += len(linkedList)

    def getLastElement(self):
        """
        Return the last element of the list.

        :return: (Node) The last element.
        """

        if self.start_node is None:
            print("List is empty")
            return None
        else:
            n = self.start_node
            while n.next is not None:
                n = n.next
            return n

    def __iter__(self):
        if self.start_node is None:
            print("List is empty")
            return None
        else:
            n = self.start_node
            while n is not None:
                yield n
                n = n.next

    def print(self):
        """
        Print the list in the right order.

        :return:
        """
        if self.start_node is None:
            print("List empty")
        else:
            n = self.start_node
            while n is not None:
                print(n.element, "-> ", end="")
                n = n.next

    def __str__(self):
        string = ""
        if self.start_node is None:
            string = "List empty"
        else:
            n = self.start_node
            while n is not None:
                string += str(n.element)
                string += " -> "
                n = n.next
        return string
