import chimera

from chimera import runCommand as rc

# Read the txt file of the chimera commands in the format of the arrangements doc
with open("path/chimera_commands.txt") as f:
    lines = f.readlines()

Bfr1_lines = lines[1::3]
Bfr2_lines = lines[2::3]

basename = "Structure6-6"  # set a base name as you want
path = "C:/Users/danist/Desktop/not_drive/Hotspots_284/simulations/"

# do we really want dimers?
# pylint: disable-next=invalid-name
def to_Chimera(blue):
    """Print the Chimera UCSF commands that generate the corresponding oligomer."""
    d = {
        10: "ob",
        9: "nh",
        2: "cv",
        3: "kg",
        1: "xm",
        0: "fr",
        11: "is",
        8: "ud",
        4: "tj",
        7: "wl",
        5: "pe",
        6: "qa",
    }

    blue_line = f"sel #1:.{':.'.join(d[b] for b in blue)}" if blue else ""
    red_line = (
        f"sel #2:.{':.'.join(v for i, v in d.items() if i not in blue)}"
        if len(blue) < 12
        else ""
    )


def generate(Bfr1_lines, Bfr2_lines, basename, path):
    for z, (line_Bfr1, line_Bfr2) in enumerate(zip(Bfr1_lines, Bfr2_lines), start=1):
        rc(line_Bfr1)
        rc("write selected #1 path/sel_Bfr1.pdb")
        rc("~select")
        rc(line_Bfr2)
        rc("write selected #2 path/sel_Bfr2.pdb")
        rc("~select")
        rc("open #31 path/sel_Bfr1.pdb")
        rc("open #32 path/sel_Bfr2.pdb")
        new_name = f"{basename}_{z}"  # add the basename numbers from 1 to the number of structures
        rc(
            f"combine #31-#32 modelId #33 name {new_name}"
        )  # combine the selections of Bfr1 and Bfr2
        rc(f"write #33 {path}{new_name}.pdb")  # saves the combined model as pdb
        rc("close #31")
        rc("close #32")
        rc(
            "molmap #33 4 onGrid #0 modelId #34"
        )  # produces a map (resolution 4 A) from the combined model
        rc(f"volume #34 save {path}{new_name}.mrc")  # saves the map
        rc("~select")
        rc("close #33")
        rc("close #34")
