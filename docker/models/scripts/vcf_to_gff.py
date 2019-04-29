import sys


def clean_line(line):
    line = line[:line.find("|") - 1]
    line_split = line.split("\t")
    if "GT" in line_split:
        line_split = line_split[:line_split.index("GT")]
    return line_split


def get_INFO_feature(info, feature):
    if feature in info:
        info = info[info.index(feature) + len(feature):]
    else:
        return "."

    if ";" in info:
        info = info[:info.index(";")]

    return info


def get_SVTYPE(svtype):
    SVTYPE_keys = {"DEL": "Deletion", "INV": "Inversion", "INS": "Insertion"}
    if svtype not in SVTYPE_keys:
        return "."
    else:
        return SVTYPE_keys[svtype]


def parse_line(line):
    line = clean_line(line)
    # print(line)
    data = ["chr" + line[0], "VCF", "", line[1], "", ".", "", "."]
    data[2] = get_SVTYPE(get_INFO_feature(line[-1], ";SVTYPE="))
    data[4] = get_INFO_feature(line[-1], ";END=")
    data[6] = get_INFO_feature(line[-1], ";SB=")
    if data[4] == "." or data[2] == "." or line[0].lower() == "x":
        return None
    return "\t".join(data)


def parse_vcf_file(file_name):
    data = open("converted_with_ins.gff", "w")
    vcf_file = open(file_name, "r").readlines()
    for line in range(len(vcf_file)):
        if "|" in vcf_file[line]:
            parsed = parse_line(vcf_file[line])
            if parsed:
                data.write(parsed)
                data.write("\n")


if __name__ == "__main__":
    file_name = sys.argv[1]
    parse_vcf_file(file_name)

