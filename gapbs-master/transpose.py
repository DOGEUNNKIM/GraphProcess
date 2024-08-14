# graph_name = "citeseer"
# graph_name = "cora"
# graph_name = "dblp"
# graph_name = "flickr"
# graph_name = "github"
# graph_name = "nell"
graph_name = "pubmed"
# graph_name = "yelp"

f_data = open("/datasets/hpca_sgcn_datasets/"+graph_name+"/"+graph_name+".el", 'r')
f_data_t = open(graph_name + "_t.el", 'w')

num_vertex = 0

transposed_graph = []

while True:
    line = f_data.readline()
    if not line: break 
    swap = [0, 0]
    graph = line.split(' ')
    swap[0] = int(graph[1])
    swap[1] = int(graph[0])
    transposed_graph.append(swap)

transposed_graph.sort()

for i in range(0, len(transposed_graph)):
    for j in range(0, 2):
        f_data_t.write(str(transposed_graph[i][j]) + " ")
    f_data_t.write("\n")