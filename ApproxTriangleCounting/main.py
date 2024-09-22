from pyspark import SparkContext, SparkConf
import sys
import os, time
import statistics
import random as rand
from CountTriangles import CountTriangles


def docs_to_edges(raw_str):
    edges = []
    for e in raw_str.split(","):
        edges.append(int(e))
    return [edges]


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
        .values()  # <----- R1 Reduce Phase
        .sum()  # <----- R1 Reduce Phase
    )

    t_final = C**2 * triangles_count  # <----- R2 Reduce Phase

    return t_final


def gather_triangles_partitions(partition):
    triangles_count_dict = {}
    for partition in partition:
        key, edges = partition[0], partition[1]
        triangles_count_dict[key] = CountTriangles(edges)
    return [(key, triangles_count_dict[key]) for key in triangles_count_dict.keys()]


def MR_ApproxTCwithSparkPartitions(edges, C):
    triangles_count = (
        edges.flatMap(lambda x: [(rand.randint(0, C - 1), x)])  # <----- R1 Map Phase
        .groupByKey()  # <----- R1 Shuffling
        .mapPartitions(gather_triangles_partitions)  # <----- R1 Reduce Phase
        .values()  # <----- R1 Reduce Phase
        .sum()  # <----- R1 Reduce Phase
    )

    t_final = C**2 * triangles_count  # <----- R1 Reduce Phase

    return t_final


def main():
    assert len(sys.argv) == 4, "Usage: python GxxxHW1.py <C> <R> <input_file_name>"

    # Read Inputs
    C = sys.argv[1]
    R = sys.argv[2]
    assert C.isdigit(), "C must be an integer"
    assert R.isdigit(), "R must be an integer"
    C, R = int(C), int(R)
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"

    # Spark Setup
    conf = SparkConf().setAppName("TriangleCounting")
    sc = SparkContext(conf=conf)

    rawData = sc.textFile(data_path, minPartitions=C).cache()

    edges = rawData.flatMap(docs_to_edges)
    number_of_edges = edges.count()

    # MR_ApproxTCwithNodeColors
    Node_Coloring_Rounds_Results = []
    Node_Coloring_Running_Time = []
    for _ in range(R):
        start = time.time()
        Count_triangle = MR_ApproxTCwithNodeColors(edges, C)
        Running_time_in_ms = (time.time() - start) * 1000
        Node_Coloring_Rounds_Results.append(Count_triangle)
        Node_Coloring_Running_Time.append(Running_time_in_ms)

    Node_Coloring_median = int(statistics.median(Node_Coloring_Rounds_Results))
    Node_Coloring_Average_time = int(statistics.mean(Node_Coloring_Running_Time))

    # MR_ApproxTCwithSparkPartitions
    start = time.time()
    Count_Triangles_SparkPartitions = MR_ApproxTCwithSparkPartitions(edges, C)
    SparkPartitions_RunningTime = int((time.time() - start) * 1000)

    print("Dataset = ", data_path)
    print("Number of Edges =", number_of_edges)
    print("Number of Colors =", C)
    print("Number of Repetitions =", R)
    print("Approximation through node coloring")
    print(f"- Number of triangles (median over {R} runs) = {Node_Coloring_median}")
    print(f"- Running time (average over {R} runs) = {Node_Coloring_Average_time} ms")
    print("Approximation through Spark partitions")
    print("- Number of triangles = ", Count_Triangles_SparkPartitions)
    print(f"- Running time = {SparkPartitions_RunningTime} ms")


if __name__ == "__main__":
    main()
