import sys

def parse_line(line, left, right):
    line_split = line.split("\t")
    line_split[0] = "chr" + line_split[0]
    line_split[1] = int(line_split[1])
    line_split[2] = int(line_split[2])
    if line_split[2] - line_split[1] >= 500:
        left_line_split = [line_split[0], str(line_split[1] - 500), str(line_split[2] + 50)]
        right_line_split = [line_split[0], str(line_split[1] - 50), str(line_split[2] + 500)]
        left.write("\t".join(left_line_split) + "\n")
        right.write("\t".join(right_line_split) + "\n")


def parse_del_file(file_name):
    first_line = True
    left = open("left_junction.txt", "w")
    right = open("right_junction.txt", "w")
    del_file = open(file_name, "r").readlines()

    for line in range(len(del_file)):
        if first_line:
            first_line = False
            continue
        parsed = parse_line(del_file[line], left, right)

    left.close()
    right.close()


if __name__ == "__main__":
    file_name = sys.argv[1]
    parse_del_file(file_name)
