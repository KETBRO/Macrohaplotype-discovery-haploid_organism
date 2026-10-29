Microhaplotypes (MH) are short genomic regions of approximately 150- 200 bp containing SNPs. IF we try to evaluate the population genomics information according to the MHs as multialleilic markers then it will increase the 
resolution power for clustering. MHs can be used for different purposes beyond population genomics. 

Here Microhaplotype_discovery.py is for discovery of MHs from the whole genome, and another one is for the selection of MHs or markers for panel design for evaluation of 
population structure.
The current script uses a 200bp sliding window approach with a 50 bp overlap. Each window must contain 3 to 10 SNPs. You can change according to your requirements.
For marker selection, the greedy and equidistance approaches have been used. 
