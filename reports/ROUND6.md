# M100 further precision

Round4 enlarged training to 131072 rows and used 131072 new holdout rows.
The fitted energy was -28.0555250874; its independent holdout energy was
-27.8018258430 +/- 0.0717821235. The unchanged primary-SE agreement
threshold was exceeded by the 0.2536992444 difference. The conservative
holdout residual 95% upper was 0.00716666 and all other declared gates
passed. This failed result remains in `reports/round4/M100/`.

In round6, all 32 corrected round4 training chains remain training; no
previous holdout is reused. Another 32 new training chains x4096 bring
training to 262144 rows. A new 32-chain x4096 holdout (seed prefix 9202)
is independent of every training chain and earlier holdouts. New banks and
checkpoints are written directly to NFS scratch. The same support,
sampling law, signed D<=1 effective operator and predeclared gates apply.
