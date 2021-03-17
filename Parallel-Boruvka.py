import numpy as np
import sys
from mpi4py import MPI


class Graph:
    """
    Graph structure

    Attributes:
        vertices_cnt: A int value of the number of vertices of a graph
        edge_cnt: A int value of the number of the edges of a graph
        local_edge_list: A list that contains edge information represented as [u, v, w],
                         the default value is []
                         The graph in each processor only have partial edge information
    """

    def __init__(self, vertices_cnt, edge_cnt, local_edge_cnt):
        """
        initialize a partial graph structure in each processor
        :param vertices_cnt: the total number of vertices of a full graph
        :param edge_cnt: the total number of edges of a full graph
        :param local_edge_cnt: the number of edges of a partial graph
        """

        self.vertices_cnt = vertices_cnt
        self.edge_cnt = edge_cnt
        # each edge has three values [u, v, w], so we use local_edge_cnt * 3
        self.local_edge_list = np.full(local_edge_cnt * 3, fill_value=-1, dtype='int')

    def find_root(self, root_list, i):
        """
        Find the root node for a given vertices. The nodes which have the same root belong to the same set.
        :param root_list: store the root node number for each node
        :param i: a given node number
        :return: the root node number
        """

        if root_list[i] == -1:
            return i
        return self.find_root(root_list, root_list[i])

    def union(self, root_list, rank_list, x, y):
        """
        A function that does union of two sets of x and y
        :param root_list: store the root node number for each node
        :param rank_list: store the rank for each node as the parent of a set
        :param x: set x
        :param y: set y
        :return:
        """

        rootx = self.find_root(root_list, x)
        rooty = self.find_root(root_list, y)

        # Combine the smaller rank set to the larger rank set
        if rank_list[rootx] < rank_list[rooty]:
            root_list[rootx] = rooty
        elif rank_list[rootx] > rank_list[rooty]:
            root_list[rooty] = rootx
        # If two sets have the same rank, merge the set which has a larger root number to the set which has a smaller
        # root number
        else:
            if rootx < rooty:
                root_list[rooty] = rootx
                rank_list[rootx] += 1
            else:
                root_list[rootx] = rooty
                rank_list[rooty] += 1


def check_size(rank, size, edge_cnt, local_edge_cnt, remain):
    """
    check if the size of processors can split the edges

    :param rank: the rank of processor
    :param size: the size of processors
    :param edge_cnt: the number of edges
    :param local_edge_cnt: the number of edges in each processor's local_edge_list
    :param remain: the number of edges in the last processor's local_edge_list if edges cannot divide by the size evenly
    :return: None
    """

    local_edge_cnt = int((edge_cnt + size - 1) / size)
    remain = edge_cnt % local_edge_cnt

    if remain != 0 and local_edge_cnt * (size - 1) + remain == edge_cnt:
        return
    elif remain == 0 and local_edge_cnt * size == edge_cnt:
        return
    else:
        if rank == 0:
            print('Error: The number of processors (size) must be able to exactly split the number of edges.')
        MPI.Finalize()
        exit(1)


def update_cheapest_edge(cheapest_edge_list, i, cheapest_edge):
    """
    update cheapest edge for node i
    :param cheapest_edge_list: Store the cheapest edge for each node
    :param i: node number
    :param cheapest_edge: edge information [u, v, w]
    :return: cheapest_edge_list
    """

    # if weight == -1, that indicates the cheapest did not set
    # if weight > new weight, replace it with a new edge
    if cheapest_edge_list[i * 3 + 2] == -1 or cheapest_edge_list[i * 3 + 2] > cheapest_edge[2]:
        cheapest_edge_list[i * 3] = cheapest_edge[0]
        cheapest_edge_list[i * 3 + 1] = cheapest_edge[1]
        cheapest_edge_list[i * 3 + 2] = cheapest_edge[2]
    return cheapest_edge_list


def merge_cheapest_edge(cheapest_edge_list, received_cheapest_edge_list):
    """
    Merge cheapest edge that received from other processor. Keep the edge with the smallest weight.
    :param cheapest_edge_list: cheapest edge list of the current processor
    :param received_cheapest_edge_list: cheapest edge list that is received from another processor.
    :return: None
    """

    vertices_cnt = int(len(cheapest_edge_list) / 3)
    for i in range(vertices_cnt):
        u, v, w = received_cheapest_edge_list[i * 3], received_cheapest_edge_list[i * 3 + 1], \
                  received_cheapest_edge_list[i * 3 + 2]
        if w != -1:
            update_cheapest_edge(cheapest_edge_list, i, [u, v, w])


def print_mst(mst):
    """
    print the edges of the mst and the total weight
    :param mst: mst structure
    :return: None
    """
    mst_weight = 0
    for i in range(mst.edge_cnt):
        u, v, w = mst.local_edge_list[i * 3], mst.local_edge_list[i * 3 + 1], mst.local_edge_list[i * 3 + 2]
        print(u, '->', v, ": ", w)
        mst_weight += w
    print('MST total weight: ', mst_weight)


def parallel_boruvka(graph, mst):
    """
    The main parallel boruvka algorithm. For each round, Each processor finds the cheapest edge list of the local graph.
    Then merge all cheapest edge lists into rank 0 and rank 0 broadcast the merged cheapest edge list to all processors
    to obtain the local minimum spanning tree.

    In the last round, the rank 0 has the ultimate cheapest edge list and the final mst.

    :param graph: the local graph in each processor
    :param mst: the final mst (shaped in rank 0)
    :return: None
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Store the root node number for each node
    root_list = [-1] * graph.vertices_cnt
    # Store the rank for each node as the parent of a set
    rank_list = [0] * graph.vertices_cnt
    # Store the cheapest edge for each node, store as (u,v,w)
    cheapest_edge_list = np.full(graph.vertices_cnt * 3, fill_value=-1, dtype='int')
    # Store the cheapest edge received from other processors
    received_cheapest_edge_list = None
    if size > 1:
        received_cheapest_edge_list = np.full(graph.vertices_cnt * 3, fill_value=-1, dtype='int')
    # Store how many edges has been merged to the MST
    current_mst_edge_cnt = 0

    while current_mst_edge_cnt <= mst.edge_cnt - 1:
        local_edge_cnt = int(len(graph.local_edge_list) / 3)
        for i in range(local_edge_cnt):
            u, v, w = graph.local_edge_list[i * 3], graph.local_edge_list[i * 3 + 1], graph.local_edge_list[i * 3 + 2]
            root1 = graph.find_root(root_list, u)
            root2 = graph.find_root(root_list, v)

            # If two nodes of current edge share the same root, the two nodes are already combined together, ignore it.
            # If not, that indicates the two set are not combined together, find the cheapest edge for those the roots.
            if root1 != root2:
                cheapest_edge_list = update_cheapest_edge(cheapest_edge_list, root1, [u, v, w])
                cheapest_edge_list = update_cheapest_edge(cheapest_edge_list, root2, [u, v, w])

        # in parallel mode, iteratively merge the cheapest_edge_list
        if size > 1:
            step = 1
            while step < size:
                if rank % (2 * step) == 0:  # the receiver processor
                    source = rank + step
                    if source < size:
                        received_cheapest_edge_list = comm.recv(source=source, tag=0)
                        merge_cheapest_edge(cheapest_edge_list, received_cheapest_edge_list)
                elif rank % step == 0:  # the sender processor
                    dest = rank - step
                    if dest >= 0:
                        comm.send(cheapest_edge_list, dest=dest, tag=0)
                step *= 2
            # root 0 broadcast the final cheapest_edge_list to all processors
            comm.Bcast(cheapest_edge_list, root=0)

        # combine the nodes for each cheapest edge
        for i in range(graph.vertices_cnt):
            if cheapest_edge_list[i * 3 + 2] != -1:
                u, v, w = cheapest_edge_list[i * 3], cheapest_edge_list[i * 3 + 1], cheapest_edge_list[i * 3 + 2]
                root1 = graph.find_root(root_list, u)
                root2 = graph.find_root(root_list, v)

                if root1 != root2:
                    # in rank 0, insert the shortest edge to the mst
                    if rank == 0:
                        mst.local_edge_list[current_mst_edge_cnt * 3] = u
                        mst.local_edge_list[current_mst_edge_cnt * 3 + 1] = v
                        mst.local_edge_list[current_mst_edge_cnt * 3 + 2] = w

                    graph.union(root_list, rank_list, root1, root2)
                    current_mst_edge_cnt += 1

        # reset the cheapest edge list
        cheapest_edge_list = np.full(graph.vertices_cnt * 3, fill_value=-1, dtype='int')


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each processor read the vertices and edges count form the first line of a csv file
    input_file = sys.argv[1]
    f = open(input_file, "r")
    first_line = f.readline().replace('\n', '').split(' ')
    vertices_cnt = int(first_line[0])
    edge_cnt = int(first_line[1])

    local_edge_cnt = int((edge_cnt + size - 1) / size)
    remain = edge_cnt % local_edge_cnt

    # check if the user assigned a valid size
    check_size(rank, size, edge_cnt, local_edge_cnt, remain)

    # Initialize the graph in each processor
    if rank == size - 1 and remain != 0:  # the last processor get the rest of edges
        local_edge_cnt = remain
    graph = Graph(vertices_cnt, edge_cnt, local_edge_cnt)
    # Initialize the MST which has full vertices but only (vertices-1) number of edges
    # The MST only used in rank 0
    mst = Graph(vertices_cnt, vertices_cnt - 1, vertices_cnt - 1)

    # Load all edges into rank 0
    full_edge_list = []
    if rank == 0:
        for lines in f.readlines():
            current_line = lines.replace('\n', '').split(' ')
            full_edge_list.append(int(current_line[0]))
            full_edge_list.append(int(current_line[1]))
            full_edge_list.append(int(current_line[2]))

        full_edge_list = np.array(full_edge_list, dtype="int")
    f.close()
    comm.Barrier()

    # Scatter the full edge list
    send_data = []
    displ = []
    if size > 1:  # parallel mode
        if rank == 0:
            send_data_cnt = local_edge_cnt * 3
            send_data_cnt_remain = remain * 3
            for i in range(size):
                # calculate for the send count and displacement
                if i == size - 1 and remain != 0:
                    send_data.append(send_data_cnt_remain)
                else:
                    send_data.append(send_data_cnt)
                if i == 0:
                    displ.append(0)
                else:
                    displ.append(displ[i - 1] + send_data[i - 1])
        comm.Scatterv([full_edge_list, send_data, displ, MPI.LONG], graph.local_edge_list, root=0)
    else:  # sequential mode
        graph.local_edge_list = full_edge_list

    start = MPI.Wtime()
    parallel_boruvka(graph, mst)
    end = MPI.Wtime()
    if rank == 0:
        print_mst(mst)
    print("Time elapse: ", end - start)


main()

