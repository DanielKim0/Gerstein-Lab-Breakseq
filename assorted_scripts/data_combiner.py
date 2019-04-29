import pandas as pd
import sys

def parse_dna_stats(file):
    """Parses DNA statistics from the last column of the breakseq results file."""
    file = open(file, "r")
    flex = []
    helix = []
    for line in file.readlines():
        data = line.split("\t")[8].split(";")
        flex.append(data[0][data[0].index("\""):-1].strip("\""))
        helix.append(data[3][data[3].index("\""):-1].strip("\""))
    return flex, helix

def parse_mech_classifications(file):
    """Parses the classifications from the breakseq results file."""
    file = open(file, "r")
    cla = []
    for line in file.readlines():
        data = line.split("\t")[8].split(";")[0]
        cla.append(data[data.index("\""):-1])
    return cla

def parse_hotspot_count(file):
    """Parses the hotspot counts from the results of the hotspot-counting script."""
    data = []
    for line in open(file, "r").readlines():
        data.append(line[0])
    return data

def parse_feat_matrix(file):
    """Parses a feature matrix, the result of the generating script that analyzes bigwig shape files."""
    data = pd.read_csv(file, sep="\t")
    data["full_seq"] = data["Chromosome"] + ":" + data["start"].apply(str) + "-" + data["end"].apply(str)
    return data.iloc[:,4:]

def parse_seq_complex(file):
    """Parses the result of the sequence complexity perl script."""
    return pd.read_csv(file, sep="\t")

def parse_intersect(file):
    """Parses the result of the TAD boundary area intersection file."""
    return pd.read_csv(file, sep="\t", header=None).iloc[:,-2:]

def remove_sames(df):
    """Removes columns with only one value throughout from a dataframe."""
    for col in df:
        if len(df[col].unique()) == 1:
            del df[col]
    return df

def create_data_matrix(file_list):
    """
    Combines all of the requisite data from the above functions to create one large matrix, and then saves it.
    """
    classifications = parse_mech_classifications(file_list[0])
    flex, helix = parse_dna_stats(file_list[1])
    hotspot_count = parse_hotspot_count(file_list[2])
    feat1 = parse_feat_matrix(file_list[3])
    feat2 = parse_feat_matrix(file_list[4])
    seq = parse_seq_complex(file_list[5])
    intersect = parse_intersect(file_list[6])

    data = feat1.iloc[:,feat1.columns != "hg38.phyloP100way.bw_mean"].merge(feat2, on="full_seq")
    data = data.merge(seq, left_on="full_seq", right_on="seq")
    data["intersect1"] = intersect.loc[:,14]
    data["intersect2"] = intersect.loc[:,15]
    data["hotspot_count"] = pd.Series(hotspot_count)
    data["flex"] = pd.Series(flex)
    data["helix"] = pd.Series(helix)
    data["classifs"] = pd.Series(classifications)

    data.pop("full_seq")
    data.pop("seq")

    data = remove_sames(data)

    data.to_csv("data.csv")

if __name__ == "__main__":
    file_list = sys.argv[1:]
    create_data_matrix(file_list)
