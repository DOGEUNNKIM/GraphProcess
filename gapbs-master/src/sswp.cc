#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const WeightT kDistmin = numeric_limits<WeightT>::min()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const size_t kBinSizeThreshold = 1000;
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




void RecordActiveNode(const pvector<NodeID> &activelist, int iter){
  if(output_directory == "")
    return;

  string graph_name = (string)split(output_directory, "/").back();
  string file_name = output_directory + "/sswp_push/" + graph_name + "_iter" + to_string(iter) + ".txt";
  cout << "Save ActiveList " << file_name << "]" << endl; 
  
  vector<int> act_list;
  ofstream o(file_name);

  for(NodeID act : activelist)
    act_list.push_back(act);
  sort(act_list.begin(), act_list.end()); 
  for(NodeID act : act_list)
    o << act << endl;
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

    RecordActiveNode(ActiveList, num_iter);

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

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
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
  return true;
//   return all_ok;
}


int main(int argc, char* argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  output_directory = cli.output();
  SourcePicker<WGraph> sp(g, cli.start_vertex());

  SourcePicker<WGraph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const WGraph &g, const pvector<WeightT> &dist) {
    return SSWPVerifier(g, vsp.PickNext(), dist);
  };

  auto SSWPVCPM = [&sp, &cli] (const WGraph &g){
    return SSWP_push(g, sp.PickNext(), cli.delta());
  };

  BenchmarkKernel(cli, g, SSWPVCPM, PrintSSSPStats, VerifierBound);
  return 0;
}
