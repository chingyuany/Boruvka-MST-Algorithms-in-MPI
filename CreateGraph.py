import random
import csv

VERTICES = 1000  # the number of vertices in the graph
RANDMIN = 0      # the min edge weight
RANDMAX = 1000   # the max edge weight

edges = []
edge_cnt = 0

for i in range(VERTICES):
    rand_cnt = VERTICES - 1 - i
    if rand_cnt == 0:
        break
    rand_arr = random.sample(range(RANDMIN, RANDMAX), rand_cnt)
    j = i + 1
    k = 0
    while j < VERTICES:
        if rand_arr[k] != 0:
            edge_cnt += 1
            edges.append([i, j, rand_arr[k]])
        j += 1
        k += 1

with open('graph.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([VERTICES] + [edge_cnt])
    for e in edges:
        writer.writerow(e)
