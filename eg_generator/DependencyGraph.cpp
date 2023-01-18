/*
 * Copyright (c) 2013 - 2015 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "DependencyGraph.h"

using namespace std;
using namespace Chakra;

DependencyGraph::DependencyGraph(const string filename)
  : graphStream(nullptr), firstWin(true), depWindowSize(4096), currID(0) {
  graphStream = new ProtoOutputStream(filename);
}

Node *DependencyGraph::addNode(NodeType node_type) {
  Node *new_node = new Node;
  uint64_t assigned_id = currID++;
  new_node->set_id(assigned_id);
  new_node->set_node_type(node_type);
  graphInfoMap[assigned_id] = new_node;
  depTrace.push_back(new_node);
  return new_node;
}

void DependencyGraph::assignDep(
    uint64_t past_node_id, uint64_t new_node_id) {
  auto past_node_iter = graphInfoMap.find(past_node_id);
  auto new_node_iter = graphInfoMap.find(new_node_id);
  assignDep(past_node_iter->second, new_node_iter->second);
}

void DependencyGraph::assignDep(Node *past_node, Node *new_node) {
  new_node->add_parent(past_node->id());
}

void DependencyGraph::writeTrace(uint32_t num_to_write) {
  depTraceItr dep_graph_itr(depTrace.begin());
  depTraceItr dep_graph_itr_start = dep_graph_itr;
  while (num_to_write > 0) {
    Node* pkt = *dep_graph_itr;
    graphStream->write(*pkt);
    delete pkt;
    dep_graph_itr++;
    num_to_write--;
  }
  depTrace.erase(dep_graph_itr_start, dep_graph_itr);
}

void DependencyGraph::flushEG() {
  // Write to graph all nodes in the depTrace.
  writeTrace(depTrace.size());
  // Delete the stream objects
  delete graphStream;
}
