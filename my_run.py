from neurips23.filter.GreatBruins.LSHANN import LSHANN

ANN = LSHANN(L=5, K=3)
ANN.fit("random-xs")
ANN.sanity_check()

