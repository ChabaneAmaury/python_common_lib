import datetime
import math
import pickle
import random
from pprint import pprint

from .shortestPath import shortestPath
import numpy as np


def getIndexOfMinValue(lst, checked):
    if not isinstance(lst, np.array.__class__):
        lst = np.array(lst)
    for v in checked:
        lst[v] = 0
    if len(lst[np.nonzero(lst)]) > 0:
        return np.where(lst == np.min(lst[np.nonzero(lst)]))[0]
    return []


def getTimeDelta(dateTime):
    if round(dateTime.minute / 10) == 1:
        complete = 10 - dateTime.minute
    else:
        complete = 0 - dateTime.minute
    if (dateTime + datetime.timedelta(minutes=complete)).hour >= 19:
        dateTime = datetime.datetime(dateTime.year, dateTime.month, dateTime.day + 1, 7, 0)
        complete = 0
    return dateTime, complete


def getNbVertices(nbEdges: int):
    """Return the result of the reciproque of the function to calculate the number of edges in a complete graph.

    :param nbEdges: (int) the number of edges (assuming it's for a complete graph)
    :return: (int) number of vertices based on the number of edges
    """
    return int((1 - np.sqrt(8 * nbEdges + 1)) / -2 + 1)


class graph:
    def __init__(self):
        """
        This class contains every tools we need to generate, import or export a graph (as a matrix), or the use it and make
        calculations on it.
        """
        self.matrix = None
        self.data = None
        self.nbVertices = None
        self.__isConnected = None

    def setMatrix(self, matrix):
        self.matrix = matrix
        self.nbVertices = len(matrix)
        self.__isConnected = None

    def generateMatrix(self, nbVertices, randomWeights=False, randomMin=0, randomMax=100):
        """
        Generate the graph's matrix based on a variety of parameters.

        :param nbVertices: (int) The total number of vertices the graph is composed of.
        :param randomWeights: (bool) Choose weather you want a complete unweighted graph or not.
        :param randomMin: (int) The minimum value to start with for the range of randomly
        generated weights (0 means no link)
        :param randomMax: (int) The maximum value to start with for the range of randomly
        generated weights (0 means no link)
        :return: None
        """
        self.__isConnected = None
        self.nbVertices = int(nbVertices)
        if randomMin > randomMax:
            randomMin, randomMax = randomMax, randomMin

        randrange = [0] + [i for i in range(randomMin, randomMax + 1)]
        p = [0.05 / (len(randrange) - 1) for _ in np.arange(len(randrange))]
        p[0] = 0.95

        if randomWeights:
            b = np.random.choice(randrange, size=(self.nbVertices, self.nbVertices), p=p)
            # b = np.random.randint(randomMin, randomMax, size=(self.nbVertices, self.nbVertices))
        else:
            b = np.random.randint(1, 2, size=(self.nbVertices, self.nbVertices))

        self.matrix = ((b + b.T) // 2)
        for i in range(self.nbVertices):
            self.matrix[i, i] = 0

    def createDataPickle(self, mongodbAPI):
        """
        Set the local data attribute the a list of traffic information. Arbitrary, we have chosen to create them in
        steps of 10 minutes, on a range of 7:00 to 19:00, and setting up busy hours from 7:00 to 9:00 and from 17:00
        to 19:00. In each data, we have the name of the corresponding edge, the coefficient to apply to the edge's
        weight, the datetime, and a random value defining if there is a problem on the road or not (accident,
        flat tire, traffic jam, etc.).

        :param mongodbAPI: (MongodbAPI) an object of type MongodbApi (defined above)
        :return: A generator containing the data generated
        """
        import datetime
        self.data = []
        nbMinutes = (19 - 7) * 60
        nbdays = 5
        busyHours = [[7, 9], [17, 19]]  # busy hours from 7:00 to 9:00 and from 17:00 to 19:00
        self.importMatrixPickle(mongodbAPI=mongodbAPI, collection="pickle_graph")

        for d in range(nbdays):
            for m in range(0, nbMinutes, 10):
                busyHourBool = False
                for r in busyHours:
                    if r[0] <= 7 + m // 60 < r[1]:
                        busyHourBool = True
                        break
                date = datetime.datetime(2020, 1, 1 + d % 30, 7 + m // 60, m % 60)
                for i in range(self.nbVertices):
                    for j in range(i):
                        if self.matrix.item(i, j) != 0:
                            self.data.append(
                                {
                                    "coefficient": np.random.randint(15,
                                                                     20) / 10 if busyHourBool else np.random.randint(10,
                                                                                                                     15) / 10,
                                    "edge": "{},{}".format(i, j),
                                    "problem_bool": np.random.randint(0, 10) if busyHourBool else np.random.randint(0,
                                                                                                                    100)
                                })
                yield date, self.data
                self.data.clear()

    def isComplete(self):
        """
        Check if the graph corresponding to the known matrix is complete. To do so, we count all the values in the
        matrix that are not equal to 0 (if so, that means the 2 vertices are not linked), and then divide it by 2 (
        symmetrical matrix). It then compare it to the result of the function for the number of edges for `n` vertices.
        If it's equal, the graph is complete, else it's not.

        :return: (Boolean) True if the graph is complete, else False
        """
        if np.count_nonzero(self.matrix > 0) // 2 == self.nbVertices * (self.nbVertices - 1) / 2:
            return True
        return False

    def importMatrixPickle(self, mongodbAPI, collection):
        """
        This method import the binary dump of the matrix and convert it back to original matrix.

        :param collection: (str) The collection
        :param mongodbAPI: (mongodbAPI) The mongoDB API
        :return: None
        """
        import pickle
        self.__isConnected = None
        for x in mongodbAPI.db[collection].find():
            self.matrix = pickle.loads(x['pickle'])
        self.nbVertices = len(self.matrix)

    def isConnected(self):
        """
        Scan the graph and to check either or not it is connected.

        :return: (bool) True is the graph is connected, False is not
        """

        if self.__isConnected is not None:
            return self.__isConnected

        visited = [False] * self.nbVertices
        parent = [0] * self.nbVertices
        neighbor = None

        def dfs(x):
            nonlocal self, parent, neighbor

            visited[x] = True
            neighbor = parent[x]
            if np.count_nonzero(self.matrix[x] > 0) > 1:
                for i in range(self.nbVertices):
                    if visited[i] == False and self.matrix.item(x, i) > 0:
                        parent[i] = x
                        neighbor = i
                        break

        while neighbor != -1:
            if neighbor is None:
                neighbor = 0
                parent[0] = -1
            dfs(neighbor)

        if self.nbVertices == sum(map(lambda x: x is True, visited)):
            self.__isConnected = True
        else:
            self.__isConnected = False

        return self.__isConnected

    def findCycleNearestNeighbor(self, start=0, limit=None, checked_vertices=None):
        """
        Worst case complexity is O(n^2).

        :param checked_vertices: (list) The already checked vertices to ignore
        :param limit: (int) The max number of vertices per cycle.
        :param start: (int) The starting vertex.
        :return: (tuple): the total cost and the path as a list
        """
        index = start
        checked_vertices = [] if checked_vertices is None else checked_vertices
        chain = []
        nb_start_checked_vertices = len(checked_vertices)
        cost = 0

        nxt = index

        while nxt is not None:
            chain.append(nxt)

            nxt = None
            for v in getIndexOfMinValue(self.matrix[start].copy(), checked_vertices):
                if v not in checked_vertices:
                    nxt = v
                    break

            if nxt is not None:
                cost += self.matrix.item(start, nxt)
                checked_vertices.append(start)
                start = nxt

            if limit is not None and len(set(checked_vertices)) >= limit + nb_start_checked_vertices:
                if self.matrix.item(index, start) > 0:
                    chain.append(index)
                return cost, chain

        if self.matrix.item(index, start) > 0:
            chain.append(index)
            # print("Cycle found!")

        return cost, chain

    def findCycleNearestNeighborXTimestamps(self, mongoDBApi, collection, start=0, limit=None, checked_vertices=None,
                                            dateTime=datetime.datetime(2020, 1, 1, 7, 0)):
        """
        Worst case complexity is O(n^2).

        :param collection: (str) The collection to use.
        :param mongoDBApi: (mongoDBApi) The mongoDB Api.
        :param dateTime: (datetime) The starting date and time.
        :param checked_vertices: (list) The already checked vertices to ignore.
        :param limit: (int) The max number of vertices per cycle.
        :param start: (int) The starting vertex.
        :return: (tuple): the total cost and the path as a list.
        """
        checked_vertices = [] if checked_vertices is None else checked_vertices
        index = start
        chain = []
        nb_start_checked_vertices = len(checked_vertices)
        cost = 0

        nxt = index

        while nxt is not None:
            chain.append(nxt)

            nxt = None

            line = self.getNewWeightLine(mongoDBApi=mongoDBApi,
                                         collection=collection,
                                         dateTime=dateTime,
                                         x=start)

            for v in getIndexOfMinValue(line, checked_vertices):
                if v not in checked_vertices:
                    nxt = v
                    break

            if nxt is not None:
                cost += line.item(nxt)
                checked_vertices.append(start)
                dateTime += datetime.timedelta(minutes=line.item(nxt))
                start = nxt

            if limit is not None and len(set(checked_vertices)) >= limit + nb_start_checked_vertices:
                if self.matrix.item(index, start) > 0:
                    chain.append(index)
                return cost, chain

        if self.matrix.item(index, start) > 0:
            chain.append(index)
            # print("Cycle found!")

        return cost, chain

    def findCycleNearestNeighborDijkstraXTimestamps(self, mongoDBApi, collection, start=0, limit=None, checked_vertices=None, dateTime=datetime.datetime(2020, 1, 1, 7, 0)):
        """
        Nearest neighbor  Worst case complexity is O(V * (V + E + V * log(E/V) * log(V))) with the optimized version of the binary heap.

        :param collection: (str) The collection to use.
        :param mongoDBApi: (mongoDBApi) The mongoDB Api.
        :param dateTime: (datetime) The starting date and time.
        :param checked_vertices: (list) The already checked vertices to ignore.
        :param limit: (int) Max number of vertices per cycle.
        :param start: (int) The starting vertex.
        :return: (tuple): the total cost and the path as a list
        """
        index = start
        chain = []
        checked_vertices = [] if checked_vertices is None else checked_vertices
        nb_start_checked_vertices = len(checked_vertices)
        cost = 0

        nxt = [index]

        while nxt is not None:
            tmp_list = [0] * self.nbVertices
            path_list = [([0], 0)] * self.nbVertices
            chain += nxt
            nxt = None

            dij = shortestPath(graph=self,
                               matrix=self.getNewWeightMatrix(mongoDBApi=mongoDBApi,
                                                              collection=collection,
                                                              dateTime=dateTime),
                               vStart=start)
            dij.dijkstraBinaryHeap2()
            for v in range(self.nbVertices):
                path_list[v] = dij.dijkstraFindPath(vEnd=v)
                tmp_list[v] = path_list[v][1]

            for v in getIndexOfMinValue(tmp_list, checked_vertices):
                nxt = v
                break

            if nxt is not None:
                cost += path_list[nxt][1]
                checked_vertices = list(set(checked_vertices + path_list[nxt][0]))
                start, nxt = nxt, path_list[nxt][0]

            if limit is not None and len(set(checked_vertices)) >= limit + nb_start_checked_vertices:
                dij = shortestPath(graph=self,
                                   matrix=self.getNewWeightMatrix(mongoDBApi=mongoDBApi,
                                                                  collection=collection,
                                                                  dateTime=dateTime),
                                   vStart=start)
                dij.dijkstraBinaryHeap2()
                t = dij.dijkstraFindPath(vEnd=index)
                chain += t[0]
                cost += t[1]
                return cost, chain

        if self.matrix.item(index, start) > 0:
            chain.append(index)
            # print("Cycle found!")

        # index = 0 if index + 1 == self.nbVertices else index + 1
        return cost, chain

    def findCycleNearestNeighborDijkstra(self, start=0, limit=None, checked_vertices=None):
        """
        Nearest neighbor  Worst case complexity is O(V * (V + E + V * log(E/V) * log(V))) with the optimized version of the binary heap.

        :param limit: (int) Max number of vertices per cycle.
        :param start: (int) The starting vertex.
        :return: (tuple): the total cost and the path as a list
        """
        index = start
        chain = []
        checked_vertices = [] if checked_vertices is None else checked_vertices
        nb_start_checked_vertices = len(checked_vertices)
        cost = 0

        # print(index)
        nxt = [index]

        while nxt is not None:
            tmp_list = [0] * self.nbVertices
            path_list = [([0], 0)] * self.nbVertices
            chain += nxt
            nxt = None
            dij = shortestPath(graph=self,
                               vStart=start)
            dij.dijkstraBinaryHeap2()
            for v in range(self.nbVertices):
                path_list[v] = dij.dijkstraFindPath(vEnd=v)
                tmp_list[v] = path_list[v][1]

            for v in getIndexOfMinValue(tmp_list, checked_vertices):
                nxt = v
                break

            if nxt is not None:
                cost += path_list[nxt][1]
                checked_vertices = list(set(checked_vertices + path_list[nxt][0]))
                start, nxt = nxt, path_list[nxt][0]

            if limit is not None and len(set(checked_vertices)) >= limit + nb_start_checked_vertices:
                dij = shortestPath(graph=self,
                                   vStart=start)
                dij.dijkstraBinaryHeap2()
                t = dij.dijkstraFindPath(vEnd=index)
                chain += t[0]
                cost += t[1]
                return cost, chain

        if self.matrix.item(index, start) > 0:
            chain.append(index)
            # print("Cycle found!")

        # index = 0 if index + 1 == self.nbVertices else index + 1
        return cost, chain

    def getNewWeightLine(self, mongoDBApi, collection, dateTime, x):
        """
        Returns the new weight corresponding to the given edge, for a specific date and time.

        :param mongoDBApi: (mongoDBApi) The mongoDB Api.
        :param collection: (str) The collection to use.
        :param dateTime: (datetime) The date and time.
        :param x: (int) The first vertex for the edge.
        :return: (numpy.array) The new line of weights based on data in the given collection.
        """
        dateTime, delta = getTimeDelta(dateTime)
        times = mongoDBApi.db[collection].find_one(
            {str(dateTime + datetime.timedelta(minutes=delta)): {'$exists': True}}
        )
        times = pickle.loads(times[str(dateTime + datetime.timedelta(minutes=delta))])
        line = self.matrix[x].copy()
        for y in times:
            y["edge"] = [int(i) for i in y["edge"].split(",")]
            if x in y["edge"]:
                y["edge"].remove(x)
                coefficient = y["coefficient"] * 2 if y["problem_bool"] == 0 else y["coefficient"]
                line[y["edge"][0]] = math.ceil(self.matrix.item(x, y["edge"][0]) * coefficient)
        return line

    def getNewWeightMatrix(self, mongoDBApi, collection, dateTime):
        """
        Returns the new weight corresponding to the given edge, for a specific date and time.

        :param mongoDBApi: (mongoDBApi) The mongoDB Api.
        :param collection: (str) The collection to use.
        :param dateTime: (datetime) The date and time.
        :return: (numpy.array) The new matrix based on data in the given collection.
        """
        dateTime, delta = getTimeDelta(dateTime)
        times = mongoDBApi.db[collection].find_one(
            {str(dateTime + datetime.timedelta(minutes=delta)): {'$exists': True}}
        )
        times = pickle.loads(times[str(dateTime + datetime.timedelta(minutes=delta))])
        matrix = self.matrix.copy()
        for item in times:
            item["edge"] = [int(i) for i in item["edge"].split(",")]
            coefficient = item["coefficient"] * 2 if item["problem_bool"] == 0 else item["coefficient"]
            matrix[item["edge"][0], item["edge"][1]] = matrix[item["edge"][1], item["edge"][0]] = math.ceil(
                self.matrix.item(item["edge"][0], item["edge"][1]) * coefficient)
        return matrix

    def bestInsertion(self, start=0):
        """
        Best insertion heuristic. Worst case complexity is O(2n^2).

        :param start: (int) The starting vertex.
        :return: (tuple): the total cost and the doubly linked list of the path
        """
        from common_libs.doublyLinkedList import doublyLinkedList
        dbList = doublyLinkedList()
        cost = 0
        dbList.insertEmptyList(start)
        vertices_to_check = [v for v in range(self.nbVertices)]
        vertices_to_check.remove(start)

        nxt = getIndexOfMinValue(self.matrix[start], [v for v in range(self.nbVertices) if v not in vertices_to_check])[
            0]
        dbList.insertAfter(start, nxt)
        vertices_to_check.remove(nxt)

        while len(vertices_to_check) > 0 and nxt is not None:
            nxt = vertices_to_check[0]
            n = dbList.start_node
            previous, minM = 0, None
            while n.next is not None:
                if self.matrix.item(n.element, nxt) > 0 and self.matrix.item(nxt, n.next.element) > 0:
                    M = self.matrix.item(n.element, nxt) + self.matrix.item(nxt, n.next.element) - self.matrix.item(
                        n.element, n.next.element)
                    if minM is None or M < minM:
                        minM, previous = M, n.element
                n = n.next
            dbList.insertAfter(previous, nxt)
            vertices_to_check.remove(nxt)

        n = dbList.start_node
        while n.next is not None:
            cost += self.matrix[n.element, n.next.element]
            n = n.next

        return cost, dbList

    def trucks(self, nbTrucks, commonVertex):
        vertices = [i for i in range(self.nbVertices)]
        vertices.remove(commonVertex)
        corres = []
        matrices = []
        p = len(vertices) // nbTrucks
        ap = len(vertices) % nbTrucks

        for i in range(0, len(vertices) - p, p):
            corres.append([commonVertex] + vertices[i:i + p])
        corres[-1] = [commonVertex] + vertices[i:i + p + ap]

        for item in corres:
            matrix = np.zeros((len(item), len(item)), dtype=int)

            for key, value in enumerate(item):
                for i in range(key):
                    matrix[key, i], matrix[i, key] = self.matrix.item(value, item[i]), self.matrix.item(value, item[i])

            matrices.append(matrix)
        return corres, matrices
