# Breakseq Docker Image

This is a docker image that runs breakseq scripts. More information (and environment variable descriptions) about breakseq can be found here: https://bioinform.github.io/breakseq2/

Note that this docker image runs both the *run_breakseq* and *breakseq2_gen_bplib* scripts; if you do not want to run one of them, comment out the command in the Dockerfile. (This also means that you do not have to set the environment variables for that command.)

The environmental variables that this script uses are those that are listed in the above link. However, as some are set for you, the list of the unset variables is as follows: REFERENCE, BAMS, WORK, BPLIB, REFERENCE_GEN, BPLIBGFF, OUTPUT.

Run the script using this format: *docker run -e REFERENCE=reference -e BAMS=bams -e WORK=work -e BPLIB=bplib -e BPLIBGFF=bplibgff -e OUTPUT=output*