#pragma once

#include <queue>
#include <unordered_map>

#include "third_party/utils/protoio.hh"
#include "eg_feeder/EGFeederNode.h"

namespace Chakra {

constexpr uint32_t EGFeederWindowSize = 4096;

class EGFeeder {
 public:
  EGFeeder(std::string filename);
  ~EGFeeder();

  void addNode(EGFeederNode* node);
  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  EGFeederNode* getNextIssuableNode();
  void pushBackIssuableNode(uint64_t node_id);
  EGFeederNode* lookupNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);

 private:
  void readNextWindow();
  template<typename T>
  void addDepsOnParent(EGFeederNode* new_node, T& dep_list);

  ProtoInputStream trace;
  uint32_t window_size;
  bool eg_complete;

  std::unordered_map<uint64_t, EGFeederNode*> dep_graph;
  std::queue<EGFeederNode*> dep_free_queue;
};

} // namespace Chakra
