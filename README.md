# Oligomer

## SageMath language

The code in this project is contained in `oligomer.py` and written in Python with references to the SageMath library. SageMath is documented at https://www.sagemath.org/.

## Executing code

1. For the time being, the recommended method for using this code is to copy-paste the contents of main.sage into the cell at https://sagecell.sagemath.org/, then add the commands you want to be executed below.
2. Alternatively, one can install SageMath as a standalone software then write Python code in a file, say `main.py`, containing the line `import oligomer`, then run `sage main`.
3. Finally one could get SageMath as a Python package via some package manager and try to `import oligomer` in standard Python code. This is not tested yet.


## Example commands

For each of the unique colourings with 6 blue dimers, up to rotations, display it as a net, as a graph, as a polyhedron, then print the corresponding Chimera code.

```python
for c in unique_colorings(6):
    c.show('net')
    c.show('graph')
    c.show('polyhedron')
    c.print_Chimera_commands()
```

For each of the unique colourings with 7 blue dimers, up to rotations, display information about its junctions, but only if it has 8 blue -> red junctions.

```python
for c in unique_colorings(7):
    if c.adjacencies.BR == 8:
        print(c)
```