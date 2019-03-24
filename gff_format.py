import sys


def parse_line(line):
    line = line.split()
    new_line = line[:8].copy()
    new_line.append(" ".join(line[8:]))
    return "\t".join(new_line)

def parse_gff_file(file_name):
    data = open("formatted.gff", "w")
    with open(file_name, "r") as gff_file:
        for line in gff_file:
            if line[0] != "#":
                data.write(parse_line(line))
                data.write("\n")
            else:
                data.write(line)

if __name__ == "__main__":
    file_name = sys.argv[1]
    parse_gff_file(file_name)
