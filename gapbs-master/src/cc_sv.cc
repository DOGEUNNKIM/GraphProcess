// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Author: Scott Beamer

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Shiloach-Vishkin [2] algorithm with
implementation optimizations from Bader et al. [1]. Michael Sutton contributed
a fix for directed graphs using the min-max swap from [3], and it also produces
more consistent performance for undirected graphs.

[1] David A Bader, Guojing Cong, and John Feo. "On the architectural
    requirements for efficient execution of graph algorithms." International
    Conference on Parallel Processing, Jul 2005.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.

[3] Kishore Kothapalli, Jyothish Soman, and P. J. Narayanan. "Fast GPU
    algorithms for graph connectivity." Workshop on Large Scale Parallel
    Processing, 2010.
*/


using namespace std;


string output_directory;

// String Split Funtion for argparse
vector<string> split(const string& str, const string& delim) {
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}




// The hooking condition (comp_u < comp_v) may not coincide with the edge's
// direction, so we use a min-max swap such that lower component IDs propagate
// independent of the edge's direction.
pvector<NodeID> ShiloachVishkin(const Graph &g) {
  pvector<NodeID> comp(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    comp[n] = n;
  bool change = true;
  int num_iter = 0;
  while (change) {
    change = false; //Active Vertices가 있을 때 까지
    num_iter++;     //iteration
    #pragma omp parallel for
    for (NodeID u=0; u < g.num_nodes(); u++) { //모든 vertex의
      for (NodeID v : g.out_neigh(u)) {        //outgoing edges들에 대하여
        NodeID comp_u = comp[u];               //Source Property
        NodeID comp_v = comp[v];               //Destination Property
        if (comp_u == comp_v) continue;        //Vertex Property가 같다면 아무것도 안하고 넘어감
        // Hooking condition so lower component ID wins independent of direction
        NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
        NodeID low_comp = comp_u + (comp_v - high_comp);
        if (high_comp == comp[high_comp]) {
          change = true;
          comp[high_comp] = low_comp;
        }
      }
    }
    #pragma omp parallel for
    for (NodeID n=0; n < g.num_nodes(); n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }
  cout << "Shiloach-Vishkin took " << num_iter << " iterations" << endl;
  return comp;
}

void RecordActiveNode(const pvector<NodeID> &activelist, int iter){
  if(output_directory == "")
    return;

  string graph_name = (string)split(output_directory, "/").back();
  string file_name = output_directory + "/cc_push/" + graph_name + "_iter" + to_string(iter) + ".txt";
  cout << "Save ActiveList " << file_name << "]" << endl; 
  
  vector<int> act_list;
  ofstream o(file_name);

  for(NodeID act : activelist)
    act_list.push_back(act);
  sort(act_list.begin(), act_list.end()); 
  for(NodeID act : act_list)
    o << act << endl;
}

pvector<NodeID> CC_push(const Graph & g) {
  pvector<NodeID> Vproperty(g.num_nodes());
  pvector<NodeID> Vtemp(g.num_nodes());
  pvector<NodeID> ActiveList(g.num_nodes());
  #pragma omp parallel for 
  for (NodeID n = 0; n < g.num_nodes(); n++) {
      Vproperty[n] = n;
      Vtemp[n] = n;
      ActiveList[n] = n;
  }

  bool change = true; 
  int num_iter = 0; 
  while (change) {
    change = false;
    cout << "[ITER" << num_iter << "] Active Vertex #: " << ActiveList.size() << endl;
    // #pragma omp parallel for 
    for (NodeID u : ActiveList) {
      for (NodeID v : g.out_neigh(u)) {
        // Process Edge & Reduce
        Vtemp[v] = min(Vproperty[u], Vtemp[v]);
      }
    }

    RecordActiveNode(ActiveList, num_iter);

    ActiveList.clear(); 
    // #pragma omp parallel for 
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      //Vtemp[n] = min(Vproperty[n], Vtemp[n]);
      if (Vproperty[n] != Vtemp[n]) {
        Vproperty[n] = Vtemp[n];
        ActiveList.push_back(n);
        change = true;
      }
    }

    num_iter++;
  }
  
  cout << "Vertex-Centric Programming Model CC(sync) took " << num_iter << " iterations" << endl;
  return Vproperty;
}


/* PULL Pseudo-Code */
// do{
//     change = false;
//     parallel_for (v : graph.vertices)
//         parallel_for (u : graph.incoming_neigh(v))
//             Vtemp[v] = Process_Edge();
    

//     parallel_for (v : graph.vertices){
//         if(Vproperty[v] != Vtemp[v]){
//             Vproperty[v] = Vtemp[v];
//             change = true;
//         }
//     }
// } while(change)


pvector<NodeID> CC_pull(const Graph & g) {
  pvector<NodeID> Vproperty(g.num_nodes());
  pvector<NodeID> Vtemp(g.num_nodes());

  #pragma omp parallel for 
  for (NodeID n = 0; n < g.num_nodes(); n++) {
      Vproperty[n] = n;
      Vtemp[n] = n;
  }

  bool change = true; 
  int num_iter = 0; 

  while (change) {
    // string file_name = "data/yelp_iter" + to_string(num_iter) + ".txt";
    // ofstream o(file_name);
    change = false;
    num_iter++;
    // #pragma omp parallel for 
    for(NodeID v=0;v<g.num_nodes();v++){
      for(NodeID u : g.in_neigh(v)){
        Vtemp[v] = min(Vproperty[u], Vtemp[v]);
      }
    }

    // #pragma omp parallel for 
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      if (Vproperty[n] != Vtemp[n]) {
        Vproperty[n] = Vtemp[n];
        //print active list to write Vproperty
        // o << n << endl;
        change = true;
      }
    }

  }

  cout << "CC_pull took " << num_iter << " iterations" << endl;
  return Vproperty;
}

void PrintCompStats(const Graph &g, const pvector<NodeID> &comp) {
  cout << endl;
  unordered_map<NodeID, NodeID> count;
  for (NodeID comp_i : comp)
    count[comp_i] += 1;
  int k = 5;
  vector<pair<NodeID, NodeID>> count_vector;
  count_vector.reserve(count.size());
  for (auto kvp : count)
    count_vector.push_back(kvp);
  vector<pair<NodeID, NodeID>> top_k = TopK(count_vector, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout << k << " biggest clusters" << endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
  cout << "There are " << count.size() << " components" << endl;
}


// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
bool CCVerifier(const Graph &g, const pvector<NodeID> &comp) {
  unordered_map<NodeID, NodeID> label_to_source;
  for (NodeID n : g.vertices())
    label_to_source[comp[n]] = n;
  Bitmap visited(g.num_nodes());
  visited.reset();
  vector<NodeID> frontier;
  frontier.reserve(g.num_nodes());
  for (auto label_source_pair : label_to_source) {
    NodeID curr_label = label_source_pair.first;
    NodeID source = label_source_pair.second;
    frontier.clear();
    frontier.push_back(source);
    visited.set_bit(source);
    for (auto it = frontier.begin(); it != frontier.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (comp[v] != curr_label)
          return false;
        if (!visited.get_bit(v)) {
          visited.set_bit(v);
          frontier.push_back(v);
        }
      }
      if (g.directed()) {
        for (NodeID v : g.in_neigh(u)) {
          if (comp[v] != curr_label)
            return false;
          if (!visited.get_bit(v)) {
            visited.set_bit(v);
            frontier.push_back(v);
          }
        }
      }
    }
  }
  for (NodeID n=0; n < g.num_nodes(); n++)
    if (!visited.get_bit(n))
      return false;
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "connected-components");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  output_directory = cli.output();
  // BenchmarkKernel(cli, g, ShiloachVishkin, PrintCompStats, CCVerifier);
  BenchmarkKernel(cli, g, CC_push, PrintCompStats, CCVerifier);
  // BenchmarkKernel(cli, g, CC_pull, PrintCompStats, CCVerifier);
  return 0;
}
