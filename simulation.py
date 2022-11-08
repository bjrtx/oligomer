from chimera import runCommand as rc

# This file is not intended to be run as a standalone Python script, but must be opened
# in Chimera and interpreted by Chimera's own Python:
# $ chimera --nogui simulation.py
# String formats are all wrong because this is Python2.7

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

    blue_line = "sel #1:.{}".format(":.".join(d[b] for b in blue)) if blue else ""
    red_line = (
        "sel #2:.{}".format(":.".join(v for i, v in d.items() if i not in blue))
        if len(blue) < 12
        else ""
    )
    return blue_line, red_line


def generate(Bfr1_lines, Bfr2_lines, basename, path):
    map_name = "284postprocess.mrc"
    Bfr1_pdb = "Bfr1_284_raz-rerefined-correction.pdb"
    Bfr2_pdb = "Bfr2_284_raz-rerefined-correction.pdb"
    rc("close session")
    rc("open " + map_name)
    rc("open " + Bfr1_pdb)
    rc("open " + Bfr2_pdb)
    rc("volume all step 1")  # not sure if useful outside graphical display?
    for idx, (line_Bfr1, line_Bfr2) in enumerate(zip(Bfr1_lines, Bfr2_lines), start=1):
        rc(line_Bfr1)
        rc("write selected #1 path/sel_Bfr1.pdb")
        rc("~select")
        rc(line_Bfr2)
        rc("write selected #2 path/sel_Bfr2.pdb")
        rc("~select")
        rc("open #31 path/sel_Bfr1.pdb")
        rc("open #32 path/sel_Bfr2.pdb")
        new_name = "{}_{}".format(basename, idx)  
        # add the basename numbers from 1 to the number of structures
        rc(
            "combine #31-#32 modelId #33 name {}".format(new_name)
        )  # combine the selections of Bfr1 and Bfr2
        rc(
            "write #33 {}{}.pdb".format(path, new_name)
        )  # saves the combined model as pdb
        rc("close #31")
        rc("close #32")
        rc(
            "molmap #33 4 onGrid #0 modelId #34"
        )  # produces a map (resolution 4 A) from the combined model
        rc("volume #34 save {}{}.mrc".format(path, new_name))  # saves the map
        rc("~select")
        rc("close #33")
        rc("close #34")


# if __name__ == "__main__":
# Read the txt file of the chimera commands in the format of the arrangements doc
with open("path/chimera_commands.txt") as f:
    lines = f.readlines()
# the first line is ignored and there is a blank line between any two entries
Bfr1_lines = lines[1::3]
Bfr2_lines = lines[2::3]
basename = "simulated" 
path = "./chimera_simulations/"
generate(Bfr1_lines, Bfr2_lines, basename, path)
