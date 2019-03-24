import sys

def count_line(line):
    line = line.lower()
    count = 0
    for i in range(len(line) - 13):
        c = line[i:i + 13]
        print(c)
        values = [c[0], c[1], c[3], c[4], c[9], c[10], c[12]]
        if all(v == "c" for v in values) and c[6] == "t":
            count += 1
    return count

def write_commands(file_name):
    results = open(sys.argv[2], "w")
    combs = False
    with open(file_name, "r") as f:
        for line in f:
            if combs:
                combs = False
                results.write(str(count_line(line)) + "\n")
            else:
                combs = True

if __name__ == "__main__":
    write_commands(sys.argv[1])

