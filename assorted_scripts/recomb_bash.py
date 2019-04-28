import sys


def parse_line(line):
    """
    Uses the results of the inputted file to create commands used to extract nucleotide strings. Used to calculate recombination hotspot motif occurrences.
    """
    line = line.split()
    return("../breakseq/samtools-0.1.19/samtools faidx ~/data/hg38.fa " + line[0] + ":" + line[1] + "-" + line[2] + " >> " + sys.argv[2] + "\n")

def write_commands(file_name):
    """
    Writes the result of the above file to a bash file for later use.
    """
    commands = open("commands.sh", "w")
    with open(file_name, "r") as f:
        for line in f:
            commands.write(parse_line(line))


if __name__ == "__main__":
    file_name = sys.argv[1]
    write_commands(file_name)
