import json, time, sys
from pprint import pprint

from .memory_profiler import profile


graph = {
    "a,b": 7,
    "a,d": 5,
    "b,c": 8,
    "b,d": 9,
    "b,e": 7,
    "c,e": 5,
    "d,e": 15,
    "d,f": 6,
    "e,f": 8,
    "e,g": 9,
    "f,g": 11
}


class Vertice:
    def __init__(self, element):
        self.element = element
        self.parent = self
        self.rank = 0


class GraphKruskal:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.result = None

    def makeSet(self, x):
        return Vertice(x)

    def find(self, x):
        if x.parent != x:
            x.parent = self.find(x.parent)
        return x.parent

    def union(self, x, y):
        xRoot = self.find(x)
        yRoot = self.find(y)
        if xRoot != yRoot:
            if xRoot.rank < yRoot.rank:
                xRoot.parent = yRoot
            else:
                yRoot.parent = xRoot
                if xRoot.rank == yRoot.rank:
                    xRoot.rank += 1

    def KruskalMST(self):
        result = []
        self.vertices = [self.makeSet(vertice) for vertice in self.vertices]
        self.edges = sorted(self.edges, key=lambda k: k[2])
        for edge in self.edges:
            if self.find(self.vertices[edge[0]]) != self.find(self.vertices[edge[1]]):
                result.append(edge)
                self.union(self.vertices[edge[0]], self.vertices[edge[1]])

        self.result = self.convertResult(result)

    def convertResult(self, result):
        return [[self.vertices[r[0]].element, self.vertices[r[1]].element, r[2]] for r in result]

    def printResult(self):
        minimumCost = 0
        if len(self.result) == len(self.vertices) - 1:
            for u, v, weight in self.result:
                minimumCost += weight
                print("{0} -- {1} == {2}".format(u, v, weight))
            print("Minimum Spanning Tree:", minimumCost)
        else:
            print("No Minimum Spanning Tree possible.")


def scanGraph(graph):
    vertices = []
    edges = []
    for key in graph.keys():
        edge = key.split(",")
        for v in edge:
            if v not in vertices:
                vertices.append(v)
        edge[0], edge[1] = vertices.index(edge[0]), vertices.index(edge[1])
        edge.append(graph[key])
        edges.append(edge)

    return vertices, edges


def scanGraphMatrix(matrix):
    vertices = []
    edges = []
    for i in range(len(matrix)):
        if i not in vertices:
            vertices.append(i)
        for j in range(i):
            if matrix.item(i, j) > 0:
                edges.append([i, j, matrix.item(i, j)])

    return vertices, edges


# @profile
def KruskalMST(graph=None):
    # Driver code
    if graph is None:
        graph = graph
    vertices, edges = scanGraph(graph)
    g = GraphKruskal(vertices, edges)
    # Function call
    g.KruskalMST()
    g.printResult()


def KruskalMSTMatrix(matrix):
    vertices, edges = scanGraphMatrix(matrix)
    g = GraphKruskal(vertices, edges)
    # Function call
    g.KruskalMST()
    g.printResult()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--inputFile", help="the JSON file containing the graph")
    args = parser.parse_args()

    if args.inputFile:
        with open(args.inputFile) as inputFile:
            graph = json.load(inputFile)

    start = time.time()
    KruskalMST(graph)

    end = time.time()

    print("##### {} seconds #####".format(end - start))
