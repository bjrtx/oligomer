# Oligomer

## SageMath language

The code in this project is contained in `oligomer.py` and written in Python with references to the SageMath library. SageMath is documented at https://www.sagemath.org/.

## Executing code

1. For the time being, the recommended method for using this code is to copy-paste the contents of `oligomer.py` into the cell at https://sagecell.sagemath.org/, then add the commands you want to be executed below.
2. Alternative method: 
    1. Install SageMath as a standalone software.
    2. Write SageMath code (in particular, almost any Python code is valid SageMath code) in a file, say `main.sage`, containing the line `import oligomer`.
    3. Place `oligomer.py` in the same directory.
    4. Run `sage main.sage`.
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

Print the Chimera commands that describe all unique colourings with 6 blue dimers, up to rotations, in a file `tmp.txt`.

```python
with open('tmp.txt', 'a') as file:
    print('Arrangements with 6 blue dimers up to rotation:\n', file=file)
    for c in unique_colorings(6):
        c.print_Chimera_commands(file=file)
````

Print each unique 6:6 colouring with its rotation class. 

````python
from collections import defaultdict
dictionary = defaultdict(list)
for c in unique_colorings(6, isomorphism=False):
    dictionary[c].append(c)
# dictionary has 48 entries, each being a list of isomorphic colourings
for c, l in dictionary.items():
    print('Canonical representation:')
    c.show('net')
    c.print_Chimera_commands()
    print('Alternate representations:')
    for alternate_colouring in l:
        alternate_colouring.show('net')
        alternate_colouring.print_Chimera_commands()
````
