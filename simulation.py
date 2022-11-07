import chimera

from chimera import runCommand as rc

# Read the txt file of the chimera commands in the format of the arrangements doc
with open("path/chimera_commands.txt") as f: 
    lines = f.readlines()

Bfr1_lines = lines[1::3]
Bfr2_lines = lines[2::3]

basename = "Structure6-6"  # set a base name as you want
path = "C:/Users/danist/Desktop/not_drive/Hotspots_284/simulations/"

def generate(Bfr1_lines, Bfr2_lines, basename, path):
    for z, (line_Bfr1, line_Bfr2) in enumerate(zip(Bfr1_lines,Bfr2_lines), start=1):
        sel_Bfr1_chains = str(line_Bfr1)
        sel_Bfr2_chains = str(line_Bfr2)
        rc(sel_Bfr1_chains)
        rc("write selected #1 path/sel_Bfr1.pdb")
        rc("~select")
        rc(sel_Bfr2_chains)
        rc("write selected #2 path/sel_Bfr2.pdb")
        rc("~select")
        rc("open #31 path/sel_Bfr1.pdb")
        rc("open #32 path/sel_Bfr2.pdb")
        new_name = f"{basename}_{z}" # add the basename numbers from 1 to the number of structures
        rc(f"combine #31-#32 modelId #33 name {new_name}") #combine the selections of Bfr1 and Bfr2
        rc(f"write #33 {path}{new_name}.pdb") #saves the combined model as pdb
        rc("close #31")
        rc("close #32")
        rc("molmap #33 4 onGrid #0 modelId #34") #produces a map (resolution 4 A) from the combined model
        rc(f"volume #34 save {path}{new_name}.mrc") # saves the map
        rc("~select")
        rc("close #33")
        rc("close #34")
