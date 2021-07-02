import sys
import numpy as np
from .priorityQueue import BinaryHeap, BucketQueue, VanEmdeBoas


class shortestPath:
    def __init__(self, graph, matrix=None, vStart=0):
        self.graph = graph
        self.matrix = matrix if matrix is not None else graph.matrix
        self.d = [np.inf] * self.graph.nbVertices
        self.d[vStart] = 0
        self.vStart = vStart
        self.previous = [np.inf] * self.graph.nbVertices

    def DialVanEmdeBoas(self):
        """
        DO NOT USE! WIP. Attempt to modify Dial's algorithm to use a Van Emde Boas priority queue. The complexity is
        reduced to O(E * log log C) where C is the max edge weight.

        :return:
        """
        Q = VanEmdeBoas(size=self.graph.nbVertices,
                        key=lambda x: x[1])
        queue = {}
        Q.insert(self.d[self.vStart])
        queue[self.d[self.vStart]] = self.vStart
        while Q.min != Q.M:
            u = Q.min
            Q.delete(u)
            u = queue[u]
            for v in np.where(self.matrix[u] > 0)[0]:
                alt = self.d[u] + self.matrix.item(u, v)
                if alt < self.d[v]:
                    # if not np.isinf(self.d[v]):
                    #     Q.delete(self.d[v])
                    self.d[v] = alt
                    self.previous[v] = u
                    Q.insert(alt)
                    queue[alt] = v

        return self.d, self.previous

    def FloydWarshall(self, oriented=False):
        """
        The main method. This method is based on the Floyd-Warshall algorithm that compute every possible distances
        between pairs of vertices. Usually, it takes only an oriented graph as input, but we modified it so we can
        iterate over an adjacency matrix for a non oriented graph. The average complexity is Θ(n^3).

        :param oriented: (bool) Define if the adjacency matrix is for an oriented graph or not
        :return:
        """
        W = self.matrix.copy()
        for k in range(self.graph.nbVertices):
            for i in range(self.graph.nbVertices):
                if oriented:
                    for j in range(self.graph.nbVertices):
                        W[i, j] = min([a for a in [W.item(i, j), W.item(i, k) + W.item(k, j)] if a > 0], default=0)
                else:
                    for j in range(i):
                        W[j, i] = W[i, j] = min([a for a in [W.item(i, j), W.item(i, k) + W.item(k, j)] if a > 0],
                                                default=0)
        return W

    def DialBucket(self):
        """
        The main method. This method buildup the parents list according to the starting vertex, using a Bucket queue
        and an optimized version of Dijkstra's algorithm (called Dial's algorithm). The worst case complexity is O(E
        + V * C) where C is the max value of an edge.

        :return:
        """
        Q = BucketQueue()
        Q.build_heap(nbBuckets=100, aList=[], max_cost=np.amax(self.matrix), key=lambda x: x[1])
        Q.insert((self.vStart, self.d[self.vStart]))
        while len(Q) > 0:
            u = Q.delete_min()[0]
            for v in np.where(self.matrix[u] > 0)[0]:
                alt = self.d[u] + self.matrix.item(u, v)
                if alt < self.d[v]:
                    Q.change_key(old=(v, self.d[v]),
                                 new=(v, alt))
                    self.d[v] = alt
                    self.previous[v] = u
        return self.d, self.previous

    def dijkstraBinaryHeap2(self):
        """
        The main method. This method buildup the parents list according to the starting vertex, using binary heaps
        and an optimized version of Dijkstra's algorithm. The average complexity is Θ(E * log(V) + V * log(n)) or Θ(
        V^2 * log(V)) (depending on the graph representation) where E is the number of edges and V the number of
        vertices. The worst case is O(E + V * log(E/V) * log(V)).

        :return:
        """
        Q = BinaryHeap()
        tmp = []
        for v in range(self.graph.nbVertices):
            if v != self.vStart:
                self.d[v] = np.inf
                self.previous[v] = None
            tmp.append((v, self.d[v]))
        Q.build_heap(tmp, key=lambda x: x[1])

        while len(Q) > 0:
            u = Q.delete_min()[0]
            for v in np.where(self.matrix[u] > 0)[0]:
                alt = self.d[u] + self.matrix.item(u, v)
                if alt < self.d[v]:
                    self.d[v] = alt
                    self.previous[v] = u
                    Q.percolate_up()
        return self.d, self.previous

    def __find_min(self, Q):
        """
        Looks for the minimal distance node in the list.

        :param Q: The list to work with.
        :return: The vertex with the minimal distance
        """
        mini, vertex = np.inf, -1
        for (v, _) in Q:
            if mini > self.d[v]:
                mini, vertex = self.d[v], v
        return vertex

    def __update_dist(self, v1, v2):
        """
        Updates the list of distances according to the newly discovered nodes, then updates the list of parents.

        :param v1: The first vertex
        :param v2: The second vertex
        :return:
        """
        if self.d[v2] > self.d[v1] + self.matrix.item(v1, v2):
            self.d[v2] = self.d[v1] + self.matrix.item(v1, v2)
            self.previous[v2] = v1

    def dijkstraBinaryHeap(self):  # O(3n^2 / 2)
        """
        The main method. This method buildup the parents list according to the starting vertex, using binary heaps.

        :return:
        """

        Q = BinaryHeap()
        Q.build_heap(aList=[(v, self.matrix.item(self.vStart, v)) for v in range(self.graph.nbVertices)],
                     key=lambda x: x[1])
        while len(Q) > 0:  # n
            v1 = Q.delete_min()[0]  # O(n/2)
            for v2, p in enumerate(self.matrix[v1]):  # n
                if p > 0:
                    self.__update_dist(v1, v2)  # 1

    def dijkstraFindPath(self, vEnd):  # O(n)
        """
        This method returns the path by iterating from the end vertex to the start vertex, through the parents list.
        The list is then completed with the starting vertex and reversed to respect the path's order.

        :param vEnd: The end vertex
        :return: The path's list.
        :return: The path's total cost.
        """
        A = []
        v = vEnd
        cost = 0
        while v != self.vStart:
            A.append(v)
            try:
                cost += self.matrix.item(v, self.previous[v])
            except TypeError:
                raise NoPathFoundException("No path found. Aborting!")
            v = self.previous[v]

        # A.append(self.vStart)
        A.reverse()
        return A, cost


class NoPathFoundException(Exception):
    """
    Exception raised when no path is found, resulting in an error.
    """
    pass


if __name__ == "__main__":
    import random
    from common_libs.graph import graph
    from pprint import pprint
    from time import time

    g = graph()
    g.generateMatrix(nbVertices=1000,
                     randomWeights=True,
                     randomMin=0,
                     randomMax=10)

    pprint(g.matrix)

    dij = shortestPath(graph=g,
                       matrix=g.matrix.copy(),
                       vStart=0)

    rand = random.randrange(g.nbVertices)
    print(rand, g.matrix.item(0, rand))

    # start_time = time()
    # dij.dijkstraBinaryHeap()
    # print(dij.dijkstraFindPath(vEnd=rand))
    # print("----------  {0} s  ----------".format(time() - start_time))

    start_time = time()
    dij.dijkstraBinaryHeap2()
    print(dij.dijkstraFindPath(vEnd=rand))
    print("----------  {0} s  ----------".format(time() - start_time))
    #
    # dij = shortestPath(graph=g, vStart=0)
    # start_time = time()
    # dij.DialBucket()
    # print(dij.dijkstraFindPath(vEnd=rand))
    # print("----------  {0} s  ----------".format(time() - start_time))

    # dij = shortestPath(graph=g, vStart=0)
    # start_time = time()
    # print(dij.FloydWarshall(oriented=False))
    # # print(dij.dijkstraFindPath(vEnd=rand))
    # print("----------  {0} s  ----------".format(time() - start_time))

    # dij = shortestPath(graph=g, vStart=0)
    # start_time = time()
    # dij.DialVanEmdeBoas()
    # print(dij.dijkstraFindPath(vEnd=rand))
    # print("----------  {0} s  ----------".format(time() - start_time))
