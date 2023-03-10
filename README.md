# Chakra
## Installation
We use `setuptools` to install/uninstall the `chakra` package:
```shell
# Install package
$ python setup.py install

# Uninstall package
$ python -m pip uninstall chakra
```

## Execution Graph Converter (eg_converter)
This tool converts execution graphs into the Chakra format.
This converter supports three types of formats: ASTRA-sim text files, FlexFlow, and PyTorch.

You can use the following commands for each input type.

### ASTRA-sim Text Files
```shell
$ python -m eg_converter.eg_converter\
    --input_type Text\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --num_dims <num_dims>\
    --num_passes <num_passes>
```

### FlexFlow Execution Graphs
```shell
$ python -m eg_converter.eg_converter\
    --input_type FlexFlow\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --npu_frequency <npu_frequency>\
    --num_dims <num_dims>
```

### PyTorch Execution Graphs
```shell
$ python -m eg_converter.eg_converter\
    --input_type PyTorch\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --default_simulated_run_time <default_simulated_run_time>\
    --num_dims <num_dims>
```

## Execution Graph Generator (eg_generator)
This is an execution graph generator that generates synthetic execution graphs.
A user can define a new function in the generator to generate new synthetic execution graphs.
You can follow the commands below to run the generator.
```shell
$ protoc eg_def.proto --proto_path eg_def --cpp_out eg_def
$ cd eg_generator/
$ cmake CMakeLists.txt && make -j$(nproc)
$ ./eg_generator  --num_npus <num_npus> --num_dims <num_dims>
```

## Execution Graph Visualizer (eg_visualizer)
This tool visualizes a given execution graph (EG) by converting the EG to a graphviz EG.
A user has to feed the output graphviz file to a graphviz visualizer such as https://dreampuf.github.io/GraphvizOnline/.

You can run this tool with the following command.
```shell
$ python -m eg_visualizer.eg_visualizer\
    --input_filename <input_filename>\
    --output_filename <output_filename>
```

## Timeline Visualizer (timeline_visualizer)
This tool visualizes the execution timeline of a given execution graph (EG).

You can run this timeline visualizer with the following command.
```shell
$ python -m timeline_visualizer.timeline_visualizer\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --npu_frequency <npu_frequency>
```

The input file is an execution trace file in csv, and the output file is a json file.
The input file format is shown below.
```csv
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
...
```
As this tool requires an execution trace of an EG, a simulator has to print out execution traces.
The output json file is chrome-tracing-compatible.
When you open the file with `chrome://tracing`, you will see an execution timeline like the one below.
![](doc/timeline_visualizer.png)

## Execution Graph Feeder (eg_feeder)
This is a graph feeder that feeds dependency-free nodes to a simulator.
Therefore, a simulator has to import this feeder as a library.
Currently, ASTRA-sim is the only simulator that supports the graph feeder.
You can run execution graphs on ASTRA-sim with the following commands.
```
$ git clone --recurse-submodules git@github.com:astra-sim/astra-sim.git
$ cd astra-sim
$ git checkout Chakra
$ git submodule update --init --recursive
$ cd extern/graph_frontend/chakra/
$ git checkout main
$ cd -
$ ./build/astra_analytical/build.sh -c

$ cd extern/graph_frontend/chakra/eg_generator
$ cmake CMakeLists.txt && make -j$(nproc)
$ ./eg_generator

$ cd -
$ ./run.sh
```
