#include "eg_feeder/EGFeeder.h"

using namespace std;
using namespace Chakra;

bool EGFeederNode::removeDep(uint64_t node_id) {
  for (auto it = parent_list.begin(); it != parent_list.end(); it++) {
    if (*it == node_id) {
      parent_list.erase(it);
      return true;
    }
  }
  return false;
}

bool EGFeederNode::removeDepOn(uint64_t node_id) {
  removeDep(node_id);
  return parent_list.empty();
}
