#include "eg_feeder/EGFeeder.h"

using namespace std;
using namespace Chakra;

EGFeeder::EGFeeder(string filename)
  : trace(filename),
  window_size(EGFeederWindowSize),
  eg_complete(false) {
  readNextWindow();
}

EGFeeder::~EGFeeder() {
}

void EGFeeder::addNode(EGFeederNode* node) {
  dep_graph[node->id()] = node;
}

void EGFeeder::removeNode(uint64_t node_id) {
  dep_graph.erase(node_id);

  if (!eg_complete
      && (dep_free_queue.size() < window_size)) {
    readNextWindow();
  }
}

bool EGFeeder::hasNodesToIssue() {
  return !(dep_graph.empty() && dep_free_queue.empty());
}

EGFeederNode* EGFeeder::getNextIssuableNode() {
  if (dep_free_queue.size() != 0) {
    EGFeederNode* node = dep_free_queue.front();
    dep_free_queue.pop();
    return node;
  } else {
    return nullptr;
  }
}

void EGFeeder::pushBackIssuableNode(uint64_t node_id) {
  EGFeederNode* node = dep_graph[node_id];
  dep_free_queue.push(node);
}

EGFeederNode* EGFeeder::lookupNode(uint64_t node_id) {
  return dep_graph[node_id];
}

void EGFeeder::freeChildrenNodes(uint64_t node_id) {
  EGFeederNode* node = dep_graph[node_id];
  for (auto child: node->children) {
    if (child->removeDepOn(node->id())) {
      dep_free_queue.push((EGFeederNode*)child);
    }
  }
}

void EGFeeder::readNextWindow() {
  uint32_t num_read = 0;
  while (num_read != window_size) {
    EGFeederNode *new_node = new EGFeederNode;

    if (!trace.read(*new_node)) {
      eg_complete = true;
      delete new_node;
      return;
    }

    for (int i = 0; i < new_node->parent_size(); i++) {
      new_node->parent_list.push_back(new_node->parent(i));
    }

    addNode(new_node);

    addDepsOnParent(new_node, new_node->parent_list);
    if (new_node->parent_list.empty()) {
      dep_free_queue.push(new_node);
    }

    num_read++;
  }
}

template<typename T>
void EGFeeder::addDepsOnParent(EGFeederNode* new_node, T& parent_list) {
  auto it = parent_list.begin();
  while (it != parent_list.end()) {
    auto parent_itr = dep_graph.find(*it);
    if (parent_itr != dep_graph.end()) {
      parent_itr->second->children.push_back(new_node);
      it++;
    } else {
      it = parent_list.erase(it);
    }
  }
}
