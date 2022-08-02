# Oligomer

## SageMath language

The code in this project is contained in main.sage and written in the SageMath language. This language is documented at https://www.sagemath.org/. It is very similar to, and based on, Python.

## Executing code

For the time being, the recommended method for using this code is to copy-paste the contents of main.sage into the cell at https://sagecell.sagemath.org/, then add the commands you want to be executed below.

## Example commands

For each of the unique colourings with 6 blue dimers, up to rotations, display it as a net, as a graph, as a polyhedron, then print the corresponding Chimera code.

```python
for c in unique_colourings(6):
    c.show('net')
    c.show('graph')
    c.show('polyhedron')
    c.print_Chimera_commands()
```

For each of the unique colourings with 7 blue dimers, up to rotations, display information about its junctions, but only if it has 8 blue -> red junctions.

```python
for c in unique_colourings(7):
    if c.adjacencies.BR == 8:
        print(c)
```
