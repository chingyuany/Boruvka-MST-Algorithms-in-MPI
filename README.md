# Boruvka-MST-Algorithms-in-MPI
A connected graph may have multiple spanning trees. Finding the minimum spanning tree
(MST) from a graph is of great significance for solving these practical application problems
This program using Boruvka’s algorithm to find the minimum spanning tree from a graph and implement it in MPI parallel programming.  

Run program: "mpirun -np {np} python Parallel-Boruvka.py {graphfile.csv}"  

Parallel-Boruvka.py:  
    Parallel version Boruvka algorithm, also run in sequential mode when running with -np 1.  

CreateGraph.py:  
    Create any graph that has distinct edge weight for each node. The output file is graph.csv
    You can run this program with python3 CreateGraph.py  

graph.csv:  
    The graph for Boruvka algorithm testing.  
