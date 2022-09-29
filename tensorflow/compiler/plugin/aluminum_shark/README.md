# HE DL Compiler

## Build options

There are a few flags that can be passed to the build using `--cxxopt=-DVARIABLE`. The following flags and variables are supported:

- `LAYOUT_DEBUG`: builds extra debug output into the layouts. It also disables parrlell execution. Making it much slower. It further tries to decrypt some intermediate results if a private key is availabe.