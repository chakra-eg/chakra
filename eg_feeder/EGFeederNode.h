#pragma once

#include <list>
#include <vector>

#include "eg_def/eg_def.pb.h"

typedef ChakraProtoMsg::Node ChakraNode;
typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::MemoryType ChakraMemoryType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

namespace Chakra {

class EGFeederNode : public ChakraProtoMsg::Node {
  public:
    bool removeDep(uint64_t dep);
    bool removeDepOn(uint64_t node_id);

    std::vector<EGFeederNode*> children;
    std::list<uint64_t> parent_list;
};

} // namespace Chakra
