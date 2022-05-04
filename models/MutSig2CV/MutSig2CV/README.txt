*********************
ALGORITHM INFORMATION
*********************

MutSig2CV analyzes somatic point mutations discovered in DNA sequencing,
identifying genes mutated more often than expected by chance given inferred
background mutation processes.  MutSig2CV consists of three independent
statistical tests, described briefly below:

Abundance (CV): The most important step for inferring genes' mutational
significance is to properly classify whether the gene is highly mutated
relative to some background mutation rate (BMR), which varies on a macroscopic
level across patients and genes and on a microscopic level across sequence
contexts.  MutSig accounts for all three of these to renormalize BMR on a
per-gene, -patient, and -context level.

Clustering (CL): Genes often harbor mutational hotspots, specific sites that
are frequently mutated.  While abundance calculations bin mutations on the gene
level, clustering bins mutations on the local site level, which allows MutSig
to differentiate between genes with uniformly distributed mutations and genes
with localized hotspots, assigning higher significance to the latter.

Conservation (FN): MutSig uses evolutionary conservation as a proxy for
determining the functional significance of a mutated site.  It assumes that
genetic sites highly conserved across vertebrates have greater functional
significance than weakly conserved sites.  MutSig assigns a higher significance
to genes that experience frequent mutations in highly conserved sites.

For detailed descriptions of the algorithms employed in the MutSig2CV suite for
each of these tests, please visit
  https://www.broadinstitute.org/cancer/cga/mutsig

Please cite:

Discovery and saturation analysis of cancer genes across 21 tumour types.
Lawrence MS, Stojanov P, Mermel CH, Robinson JT, Garraway LA, Golub TR, 
Meyerson M, Gabriel SB, Lander ES, Getz G.
Nature. 2014 Jan 23;505(7484):495-501. doi: 10.1038/nature12912.

***************
RUN INFORMATION
***************

MutSig2CV was run on all TCGA cohorts (indicated by disease code) and on all
cohorts aggregated together (PANCAN).  We recommend selecting genes with 
q-value <= 0.1
