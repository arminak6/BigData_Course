from pyspark import SparkContext, SparkConf
import sys
import os, time
import statistics
import random as rand
from collections import defaultdict


# Count Triangles2
def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    # We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)
    # Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    # Creare a dictionary for storing node colors
    node_colors = dict()

    for edge in edges:
        u, v = edge
        node_colors[u] = ((rand_a * u + rand_b) % p) % num_colors
        node_colors[v] = ((rand_a * v + rand_b) % p) % num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted(
                            (node_colors[u], node_colors[v], node_colors[w])
                        )
                        # If triangle has the right colors, count it.
                        if colors == triangle_colors:
                            triangle_count += 1

    # Return the total number of triangles in the graph
    return triangle_count


# Count Triangles
def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex

    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


# Raw Str to Edges
def docs_to_edges(raw_str):
    edges = []
    for e in raw_str.split(","):
        edges.append(int(e))
    return [edges]



# MR_ApproxTCwithNodeColors
def MR_ApproxTCwithNodeColors(edges, C):
    P = 8191
    a, b = rand.randint(1, P - 1), rand.randint(0, P - 1)

    def hash_func(edge):
        vertex1, vertex2 = edge[0], edge[1]
        hash_vertex1 = ((a * vertex1 + b) % P) % C
        hash_vertex2 = ((a * vertex2 + b) % P) % C

        if hash_vertex1 == hash_vertex2:
            return [(hash_vertex1, [vertex1, vertex2])]
        return []

    triangles_count = (
        edges.flatMap(hash_func)  # <----- R1 Map Phase
        .groupByKey()  # <----- R1 Shuffling
        .mapValues(CountTriangles)  # <----- R1 Reduce Phase
        .values()  # <----- R2 Reduce Phase
        .sum()  # <----- R2 Reduce Phase
    )

    t_final = C**2 * triangles_count  # <----- R2 Reduce Phase

    return t_final


# MR_ExactTC
def MR_ExactTC(edges, C):
    P = 8191
    a, b = rand.randint(1, P - 1), rand.randint(0, P - 1)
    # color_nodes = return_color_nodes(C)

    def hash_func2(edge):
        key_value_pairs = []
        vertex1, vertex2 = edge[0], edge[1]
        hash_vertex1 = ((a * vertex1 + b) % P) % C
        hash_vertex2 = ((a * vertex2 + b) % P) % C

        for i in range(C):
            sorted_key = tuple(sorted((hash_vertex1, hash_vertex2, i)))
            key_value_pairs.append([sorted_key, (vertex1, vertex2)])

        return key_value_pairs


    def count_triangles(partition):
        triangles_counted = []
        triangles_count = countTriangles2(partition[0], partition[1], a, b, P, C)
        triangles_counted.append((partition[0], triangles_count))
        return triangles_counted

    
    triangles_count = (
        edges.flatMap(hash_func2)  # <----- R1 Map Phase
        .groupByKey()  # <----- R1 Shuffling
        .map(count_triangles)  # <----- R1 Reduce Phase
        .map(lambda item: [0, item[0][1]])  # <----- R2 Shuffling
        .values()  # <----- R2 Reduce Phase
        .sum()  # <----- R2 Reduce Phase
    )

    return triangles_count


def main():
    if len(sys.argv) != 5:
        print("Usage: python GxxxHW2.py <C> <R> <F> <input_file_name>")
        sys.exit(-1)

    # Read Inputs
    C = sys.argv[1]
    R = sys.argv[2]
    F = sys.argv[3]
    if C.isdigit() == False:
        print("C must be an integer")
        sys.exit(-1)

    if R.isdigit() == False:
        print("R must be an integer")
        sys.exit(-1)

    if F.isdigit() == False or (int(F) != 1 and int(F) != 0):
        print("F must be an integer and should only have values 0 or 1 ")
        sys.exit(-1)

    C, R, F = int(C), int(R), int(F)
    data_path = sys.argv[4]
    if os.path.isfile(data_path) == False:
        print("File or folder not found")
        sys.exit(-1)

    # Spark Setup
    conf = SparkConf().setAppName("TriangleCounting")
    sc = SparkContext(conf=conf)
    conf.set("spark.locality.wait", "0s")

    rawData = sc.textFile(data_path, minPartitions=C).cache()

    edges = rawData.flatMap(docs_to_edges)
    edges = edges.repartition(numPartitions=32)
    number_of_edges = edges.count()

    # MR_ApproxTCwithNodeColors
    if F == 0:
        Node_Coloring_Rounds_Results = []
        Node_Coloring_Running_Time = []
        for _ in range(R):
            start = time.time()
            Node_Coloring_Count_Triangle = MR_ApproxTCwithNodeColors(edges, C)
            Running_time_in_ms = (time.time() - start) * 1000
            Node_Coloring_Rounds_Results.append(Node_Coloring_Count_Triangle)
            Node_Coloring_Running_Time.append(Running_time_in_ms)

        Node_Coloring_median = int(statistics.median(Node_Coloring_Rounds_Results))
        avg_time = int(statistics.mean(Node_Coloring_Running_Time))

    # MR_ExactTC
    if F == 1:
        Exact_Coloring_Running_Time = []
        for _ in range(R):
            start = time.time()
            Exact_Count_Triangle = MR_ExactTC(edges, C)
            Running_time_in_ms = (time.time() - start) * 1000
            Exact_Coloring_Running_Time.append(Running_time_in_ms)

        Exact_Coloring_Average_time = int(statistics.mean(Exact_Coloring_Running_Time))

    print("Dataset = ", data_path)
    print("Number of Edges =", number_of_edges)
    print("Number of Colors =", C)
    print("Number of Repetitions =", R)
    if F == 0:
        print("Approximation through node coloring")
        print(f"- Number of triangles (median over {R} runs) = {Node_Coloring_median}")
        print(f"- Running time (average over {R} runs) = {avg_time} ms")
    else:
        print("Approximation through Exact node coloring")
        print("- Number of triangles = ", Exact_Count_Triangle)
        print(f"- Running time = {Exact_Coloring_Average_time} ms")


if __name__ == "__main__":
    main()
