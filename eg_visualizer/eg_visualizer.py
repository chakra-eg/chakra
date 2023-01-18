#!/usr/bin/env python3

import argparse

from third_party.utils.protolib import openFileRd, decodeMessage
from eg_def.eg_def_pb2 import *

def main():
    parser = argparse.ArgumentParser(
            description="Execution Graph Visualizer"
    )
    parser.add_argument(
            "--input_filename",
            type=str,
            default=None,
            required=True,
            help="Input Chakra execution graph filename"
    )
    parser.add_argument(
            "--output_filename",
            type=str,
            default=None,
            required=True,
            help="Output Graphviz graph filename"
    )
    args = parser.parse_args()

    chakra_eg = openFileRd(args.input_filename)
    node = Node()
    with open(args.output_filename, "w") as f:
        f.write("digraph taskgraph {\n")
        while decodeMessage(chakra_eg, node):
            label = "%s" % (node.name)
            f.write("  node%d [label=\"{%s}\", shape=\"record\"]\n"\
                    % (node.id, label))
            for parent_id in node.parent:
                f.write("  node%d->node%d\n" % (parent_id, node.id))
        f.write("}")
    chakra_eg.close()

if __name__ == "__main__":
    main()
