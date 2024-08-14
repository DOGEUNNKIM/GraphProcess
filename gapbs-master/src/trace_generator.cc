#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <limits>
#include <queue>




#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"

#include <string>
#include "platform_atomics.h"
#include "sliding_queue.h"


// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

#define MODE 0

string output_directory;
const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const WeightT kDistmin = numeric_limits<WeightT>::min()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const size_t kBinSizeThreshold = 1000;

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


int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    if (parent[u] < 0) {
      for (NodeID v : g.in_neigh(u)) {
        if (front.get_bit(v)) {
          parent[u] = v;
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  return awake_count;
}


int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for reduction(+ : scout_count) nowait
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter; //check the child of previous iter NODEs
      for (NodeID v : g.out_neigh(u)) {
        NodeID curr_val = parent[v]; //check the flag if the child has parent
        if (curr_val < 0) {  // if there's no parent
          if (compare_and_swap(parent[v], curr_val, u)) { //flag u (v's parent = u)
            lqueue.push_back(v); //push to the queue for Next Iteration
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for nowait
    for (NodeID n=0; n < g.num_nodes(); n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1; // -1 why?? 
  return parent;
}

void RecordActiveNodeQ(const SlidingQueue<NodeID> &queue, int iter){
  if(output_directory == "")
    return;
  string graph_name = (string)split(output_directory, "/").back();
  string file_name = output_directory + "/bfs_push/" + graph_name + "_iter" + to_string(iter) + ".txt";
  cout << "Save ActiveList " << file_name << "]" << endl; 
  vector<int> act_list;
  ofstream o(file_name);
  
  for(NodeID act : queue)
    act_list.push_back(act);
  sort(act_list.begin(), act_list.end()); 
  for(NodeID act : act_list)
    o << act << endl;
}

void RecordActiveNode(const pvector<NodeID> &activelist, int iter, string algorithm){
  if(output_directory == "")
    return;

  string graph_name = (string)split(output_directory, "/").back();
  string file_name = output_directory + algorithm + graph_name + "_iter" + to_string(iter) + ".txt";
  cout << "Save ActiveList " << file_name << "]" << endl; 
//   mkdir(output_directory+algorithm, 0700);
  
  vector<int> act_list;
  ofstream o(file_name);

  for(NodeID act : activelist)
    act_list.push_back(act);
  sort(act_list.begin(), act_list.end()); 
  for(NodeID act : act_list)
    o << act << endl;
}


pvector<NodeID> DOBFS(const Graph &g, NodeID source, int alpha = 1,
                      int beta = 18) {
  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  int iter = 0;
  t.Start();
  pvector<NodeID> parent = InitParent(g);
  t.Stop();
  PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        // RecordUnvisitedNode(parent, iter);
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
        PrintStep("bu", t.Seconds(), awake_count);
        iter++;
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      RecordActiveNodeQ(queue, iter);
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue);
      queue.slide_window();
      t.Stop();
      PrintStep("td", t.Seconds(), queue.size());
      iter++;
    }
  }
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
  return parent;
}


void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
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
    // #pragma omp parallel for 
    for (NodeID u : ActiveList) {
      for (NodeID v : g.out_neigh(u)) {
        // Process Edge & Reduce
        Vtemp[v] = min(Vproperty[u], Vtemp[v]);
      }
    }

    RecordActiveNode(ActiveList, num_iter, "/cc_push/");

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


pvector<WeightT> SSSP_push(const WGraph &g, NodeID source, WeightT delta) {
  pvector<WeightT> Vproperty(g.num_nodes(), kDistInf);
  pvector<WeightT> Vtemp(g.num_nodes(), kDistInf);
  pvector<NodeID> ActiveList;

  Vproperty[source] = 0;
  ActiveList.push_back(source);
  cout << "SOURCE: " << source << endl;
  bool change = true; 
  int num_iter = 0; 
  while (change) {
    change = false;
    cout << "[ITER" << num_iter << "] Active Vertex #: " << ActiveList.size() << endl;
    // #pragma omp parallel for 
    for (NodeID u : ActiveList) {
      for (WNode wv : g.out_neigh(u)) {
        // Process Edge
        WeightT edgeProResult = Vproperty[u] + wv.w; 
        
        // Reduce
        Vtemp[wv.v] = min(edgeProResult, Vtemp[wv.v]);
      }
    }

    RecordActiveNode(ActiveList, num_iter, "/sssp_push/");

    ActiveList.clear(); 
    // #pragma omp parallel for 
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      // Apply
      WeightT applyRes = min(Vproperty[n], Vtemp[n]);
      
      if (Vproperty[n] != applyRes) {
        Vproperty[n] = Vtemp[n];
        ActiveList.push_back(n);
        change = true;
      }
    }

    num_iter++;
  }
  
  cout << "Vertex-Centric Programming Model SSSP(sync) took " << num_iter << " iterations" << endl;
  return Vproperty;
}



void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}



pvector<WeightT> SSWP_push(const WGraph &g, NodeID source, WeightT delta) {
  pvector<WeightT> Vproperty(g.num_nodes(), kDistmin);
  pvector<WeightT> Vtemp(g.num_nodes(), kDistmin);
  pvector<NodeID> ActiveList;

  Vproperty[source] = kDistInf;
  ActiveList.push_back(source);
  
  bool change = true; 
  int num_iter = 0; 
  cout << "SOURCE: " << source << endl;
  while (change) {
    change = false;
    cout << "[ITER" << num_iter << "] Active Vertex #: " << ActiveList.size() << endl;
    // #pragma omp parallel for 
    for (NodeID u : ActiveList) {
      for (WNode wv : g.out_neigh(u)) {
        // Process Edge
        WeightT edgeProResult = min(Vproperty[u], wv.w); 
        
        // Reduce
        Vtemp[wv.v] = max(edgeProResult, Vtemp[wv.v]);
      }
    }

    RecordActiveNode(ActiveList, num_iter, "/sswp_push/");

    ActiveList.clear(); 
    // #pragma omp parallel for 
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      // Apply
      WeightT applyRes = max(Vproperty[n], Vtemp[n]);
      
      if (Vproperty[n] != applyRes) {
        Vproperty[n] = Vtemp[n];
        ActiveList.push_back(n);
        change = true;
      }
    }

    num_iter++;
  }
  
  cout << "Vertex-Centric Programming Model SSWP(sync) took " << num_iter << " iterations" << endl;
  return Vproperty;
}

void PrintSSWPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSWP Tree reaches " << num_reached << " nodes" << endl;
}


// Compares against simple serial implementation
bool SSWPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistmin);
  oracle_dist[source] = kDistInf;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (min(td, wn.w) > oracle_dist[wn.v]) {
          oracle_dist[wn.v] = min(td, wn.w);
          mq.push(make_pair(min(td, wn.w), wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
    //   cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }

  //TODOOOOOOOOOOOO, VERIFY THE FUNCTIONALITY IF POSSIBLE
  //BUT this must be correct
  all_ok = true;
  return all_ok;
}



int main(int argc, char* argv[]) {
  bool change = false;

  if(change){
    CLApp cli(argc, argv, "trace generator for bfs, cc");
    if (!cli.ParseArgs())
        return -1;

    cout << "bfs cc" << endl;
    output_directory = cli.output();
    Builder b(cli);
    Graph g = b.MakeGraph();
    SourcePicker<Graph> sp(g, cli.start_vertex());
    auto BFSBound = [&sp] (const Graph &g) { return DOBFS(g, sp.PickNext()); };
    SourcePicker<Graph> vsp(g, cli.start_vertex());
    auto VerifierBound = [&vsp] (const Graph &g, const pvector<NodeID> &parent) {
        return BFSVerifier(g, vsp.PickNext(), parent);
    };
    BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
    BenchmarkKernel(cli, g, CC_push, PrintCompStats, CCVerifier);

  }
  else{
    cout << "sssp, sswp" << endl;
    
    CLDelta<WeightT> wcli(argc, argv, "trace generator for sssp, sswp");
    if (!wcli.ParseArgs())
        return -1;
    output_directory = wcli.output();
    WeightedBuilder wb(wcli);
    WGraph wg = wb.MakeGraph();
    SourcePicker<WGraph> sp(wg, wcli.start_vertex());
    SourcePicker<WGraph> vsp(wg, wcli.start_vertex());
    
    auto VerifierBounds = [&vsp] (const WGraph &wg, const pvector<WeightT> &dist) {
        return SSSPVerifier(wg, vsp.PickNext(), dist);
    };
    auto SSSPVCPM = [&sp, &wcli] (const WGraph &wg){
        return SSSP_push(wg, sp.PickNext(), wcli.delta());
    };
    BenchmarkKernel(wcli, wg, SSSPVCPM, PrintSSSPStats, VerifierBounds);

    auto VerifierBoundw = [&vsp] (const WGraph &wg, const pvector<WeightT> &dist) {
        return SSWPVerifier(wg, vsp.PickNext(), dist);
    };
    auto SSWPVCPM = [&sp, &wcli] (const WGraph &wg){
        return SSWP_push(wg, sp.PickNext(), wcli.delta());
    };
    BenchmarkKernel(wcli, wg, SSWPVCPM, PrintSSWPStats, VerifierBoundw);
  }
  
  return 0;
}
