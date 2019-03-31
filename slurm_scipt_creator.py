import sys

def argv_length_error():
    print("7 Total Args")
    print("Arg 1: Batch file name")
    print("Arg 2: Use Gerstein partition?")
    print("Arg 3: Memory (MB)")
    print("Arg 4: Time (days)")
    print("Arg 5: Number of CPUs")
    print("Arg 6: Python file")
    print("Arg 7: Data file")
    print()

def create_slurm(args):
    output = open(args[0], "r")
    output.write("#!/bin/bash\n")
    if args[1] > 0:
        output.write("#SBATCH --partition=pi_gerstein\n")
    output.write("#SBATCH --mem " + str(args[2]) + "\n")
    output.write("#SBATCH -t " + str(args[3]) + "- #days")
    output.write("#SBATCH -c " + str(args[4]) + "\n")
    output.write("source activate mlenv")
    output.write("python " + str(args[5]) + "  " + str(args[6]) + "\n")
    output.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        argv_length_error()
    else:
        create_slurm(sys.argv[1:])