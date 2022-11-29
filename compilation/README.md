# On-Device Training Under 256KB Memory
This section shows how we translate a pytorch models into intermediate representations (IR), and how to transform this forward IR to training version and perform preprocessing.

Pre-requests:
1. Setup forked tvm  [third_party/tvm-hack](https://github.com/lyken17/tvm-hack)
    * Pull the remove folder via `git submodule update --init --recursive`
    * Install GCC/Clang, LLVM (<14.0) and Python (<3.10)
    * Compile following [tvm compile-from-source](https://tvm.apache.org/docs/install/from_source.html) (enable LLVM during compilation)
    * Export the compiled TVM in path
        ```
        export TVM_HOME=<DIR to third_party/tvm-hack>
        export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
        ```
2. Translate pytorch models into training IR
    ```
    python mcu_ir_gen.py 
    ```
    and it will generate all required IRs and information under `ir_zoos`.
3. Convert the IR into a json format to enable MCU integration
    ```
    python ir2json.py <target IR path>
    ```
    and the json files will be stored in current directory.