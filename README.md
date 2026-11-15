# Clustering from Scratch in NumPy: HAC, K-Means++, Medoids, GMMs and Spectral Cuts

Build every clustering method in Murphy's Probabilistic Machine Learning in NumPy, run each on the data behind the book's figures, and find out which one to trust. Score clusterings with purity, the Rand index, adjusted Rand and NMI, reproducing the 17-object example exactly. Grow dendrograms on real yeast gene expression and watch single link chain 235 of 310 genes into one cluster while complete link stays compact. Implement Lloyd's algorithm and K-means++ and land on the book's distortion of 2800.54. Compress an image with vector quantization and see the codebook cost half the bits. Cluster DNA reads by edit distance with K-medoids, where no mean exists. Fit Gaussian mixtures with four covariance types, prove on the data that hard EM is K-means, and pick K with BIC while validation distortion only ever asks for more clusters. Then cut graphs: spectral clustering recovers rings and spirals that K-means cannot, as long as you get the kernel width right. A final bake-off runs nine methods on eight datasets, and no method wins them all.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** pairwise_distances
- [x] **2.** contingency_table
- [x] **3.** pair_counts
- [x] **4.** cluster_scores
- [x] **5.** agglomerate
- [x] **6.** cut_tree
- [x] **7.** single_link_mst
- [x] **8.** same_hierarchy
- [x] **9.** linkage_report
- [x] **10.** kmeans_lloyd
- [x] **11.** kmeanspp_init
- [x] **12.** kmeans_restarts
- [x] **13.** vq_bits
- [x] **14.** block_vq
- [x] **15.** edit_distance_matrix
- [x] **16.** kmedoids_voronoi
- [x] **17.** pam
- [x] **18.** gmm_e_step
- [x] **19.** gmm_m_step
- [x] **20.** fit_gmm
- [x] **21.** em_kmeans_limit
- [x] **22.** bic_score
- [x] **23.** silhouette_samples
- [x] **24.** k_selection_curves
- [x] **25.** gaussian_affinity
- [x] **26.** graph_laplacians
- [x] **27.** spectral_clustering
- [x] **28.** ncut_value
- [x] **29.** fit_bernoulli_mixture
- [x] **30.** clustering_bakeoff

## Results

```
== Part 1. Scoring a clustering (Figure 21.1, 17 objects) ==
  contingency table [[5, 1, 0], [1, 4, 1], [2, 0, 3]] pairs (TP, FP, FN, TN) (20, 20, 24, 72)
  clustering   {'purity': 0.706, 'rand': 0.676, 'ari': 0.243, 'mi': 0.392, 'nmi': 0.365}
  singletons   {'purity': 1.0, 'rand': 0.676, 'ari': 0.0, 'mi': 1.055, 'nmi': 0.543} (purity 1 and ARI 0)

== Part 2. Hierarchical clustering ==
  Figure 21.2 single-link heights: [1.0, 1.0, 2.5, 3.0]
  Figure 21.4, 310 yeast genes, correlation distance, cut into 16 clusters:
    single    top height 0.325  largest cluster 235  singletons 14  widest cluster 1.112
    complete  top height 1.999  largest cluster  87  singletons  2  widest cluster 0.374
    average   top height 1.830  largest cluster 201  singletons  5  widest cluster 0.586
  Same tree after squaring the distances? {'single': True, 'complete': True, 'average': False}
  1000 points: agglomerate 2.94s, MST route 0.037s, same heights: True

== Part 3. K-means ==
  book data K=4, 200 starts: random mean 2848.2 best 2800.54, K-means++ mean 2853.0 best 2800.54
  book data K=5, 200 starts: random mean 2138.1 best 2082.32, K-means++ mean 2128.6 best 2082.32
    local optima at K=4 (random starts): [2800.5, 2809.6, 2978.1] (book: 2800.54 and 2978.45)
  25 tight clusters, data seed 0: random mean 1378 best 460.8, K-means++ mean 499 best 168.4
  25 tight clusters, data seed 1: random mean 1498 best 790.0, K-means++ mean 484 best 178.5
  25 tight clusters, data seed 2: random mean 1481 best 479.6, K-means++ mean 401 best 169.5

== Part 4. Vector quantization (200 x 320 grey image, 512,000 raw bits) ==
  1x1 blocks K=  2: codes 1.000 bpp, total 1.000 bpp, PSNR 17.80 dB
  1x1 blocks K=  4: codes 2.000 bpp, total 2.001 bpp, PSNR 23.26 dB
  1x1 blocks K= 16: codes 4.000 bpp, total 4.002 bpp, PSNR 35.28 dB
  2x2 blocks K= 16: codes 1.000 bpp, total 1.008 bpp, PSNR 22.85 dB
  2x2 blocks K=256: codes 2.000 bpp, total 2.128 bpp, PSNR 29.66 dB
  4x4 blocks K= 16: codes 0.250 bpp, total 0.282 bpp, PSNR 20.03 dB
  4x4 blocks K=256: codes 0.500 bpp, total 1.012 bpp, PSNR 23.37 dB
  5x5 blocks K=256: codes 0.320 bpp, total 1.120 bpp, PSNR 22.90 dB
  8x8 blocks K=256: codes 0.125 bpp, total 2.173 bpp, PSNR 23.79 dB

== Part 5. K-medoids ==
  DNA read families, 20 random starts each (no feature vectors, only edit distance):
    data seed 0: Voronoi cost 782.4 ARI 0.68; PAM cost 683.0 (range 683-683) ARI 0.90; HAC {'complete': 0.78, 'average': 0.88}
    data seed 1: Voronoi cost 758.6 ARI 0.79; PAM cost 673.0 (range 673-673) ARI 0.97; HAC {'complete': 0.97, 'average': 0.99}
    data seed 2: Voronoi cost 812.5 ARI 0.67; PAM cost 709.0 (range 709-709) ARI 0.94; HAC {'complete': 0.89, 'average': 0.83}
  Three blobs plus far outliers, mean ARI on the blobs over 5 seeds:
     0 outliers: K-means 0.99, K-medoids 0.99
     1 outliers: K-means 0.56, K-medoids 0.99
     3 outliers: K-means 0.11, K-medoids 0.99
     6 outliers: K-means 0.00, K-medoids 0.99
    12 outliers: K-means 0.00, K-medoids 0.99

== Part 6. Gaussian mixtures (book data, K = 5, Figure 21.14) ==
  full      ll/N -3.287  params 29  BIC -6684.4  ARI 0.973
  tied      ll/N -3.458  params 17  BIC -6979.8  ARI 0.902
  diag      ll/N -3.489  params 24  BIC -7068.3  ARI 0.598
  spherical ll/N -3.489  params 19  BIC -7050.5  ARI 0.595
  K-means   ARI 0.684
  Hard EM vs Lloyd from one start: same labels True, largest mean difference 1.1e-12
    soft EM sigma = 5.00: mean difference 2.6502, labels differ 0.6345, distortion 13104
    soft EM sigma = 1.00: mean difference 1.6940, labels differ 0.2785, distortion 2250
    soft EM sigma = 0.50: mean difference 0.1049, labels differ 0.0150, distortion 2185
    soft EM sigma = 0.10: mean difference 0.0014, labels differ 0.0000, distortion 2183
    soft EM sigma = 0.05: mean difference 0.0001, labels differ 0.0000, distortion 2183
  BIC over K = 2..8, higher is better:
    full      [-7397, -7145, -6770, -6684, -6704, -6725, -6747] picks K = 5
    tied      [-7688, -7489, -7182, -6980, -6916, -6831, -6802] picks K = 8
    diag      [-7410, -7153, -7066, -7069, -7024, -6984, -6928] picks K = 8
    spherical [-7928, -7349, -7055, -7051, -7030, -6963, -6915] picks K = 8

== Part 7. Choosing K (Figures 21.11 and 21.13) ==
  K               [2, 3, 4, 5, 6, 7, 8]
  val distortion  [3.62, 1.87, 1.44, 1.08, 0.82, 0.62, 0.54]
  BIC             [-3707, -3521, -3385, -3351, -3368, -3382, -3401]
  silhouette      [0.424, 0.516, 0.48, 0.476, 0.523, 0.534, 0.522]
  picks           {'val_distortion': 8, 'bic': 5, 'silhouette': 7} (true K = 5)
  all 2000 points, silhouette at K = 3..7: [0.5153, 0.4839, 0.4712, 0.5009, 0.5194]
  20 generated datasets per shape with true K in 2..6, hits out of 20:
    spherical {'val_distortion': 0, 'bic': 19, 'silhouette': 17}
    elongated {'val_distortion': 0, 'bic': 18, 'silhouette': 15}

== Part 8. Spectral clustering (Figure 21.19) ==
  rings: K-means 0.006, full GMM 0.006, complete link 0.003, single link 1.000
    spectral ARI by sigma: {0.02: 0.309, 0.05: 1.0, 0.1: 1.0, 0.2: 1.0, 0.3: 0.487, 0.5: 0.009, 1.0: 0.007, 2.0: 0.006}
  spirals: K-means 0.001, full GMM 0.001, complete link 0.000, single link 1.000
    spectral ARI by sigma: {0.02: 0.105, 0.05: 1.0, 0.1: 1.0, 0.2: 1.0, 0.3: 1.0, 0.5: 1.0, 1.0: 0.001, 2.0: 0.001}
  Two triangles + pendant: min cut [1, 1, 1, 1, 1, 1, 0] value 0.020; min Ncut [1, 1, 1, 0, 0, 0, 0] value 0.00824
  Random-walk check: p(out|S) + p(S|out) = 0.01647 = 2 x Ncut of eq 21.29 = 0.01647

== Part 9. Binary data and the bake-off ==
  digits K=10, 5 seeds, ARI/NMI/purity: Bernoulli mixture [0.569, 0.705, 0.725] (ARI sd 0.022), K-means on bits [0.578, 0.672, 0.728] (ARI sd 0.029), K-means on grey [0.642, 0.737, 0.778]
  digits K=20, 5 seeds, ARI/NMI/purity: Bernoulli mixture [0.534, 0.718, 0.864] (ARI sd 0.028), K-means on bits [0.514, 0.697, 0.842] (ARI sd 0.011), K-means on grey [0.609, 0.769, 0.917]
  Bernoulli mixture BIC by K: {5: -39643, 10: -36966, 20: -36644, 30: -37842, 40: -39338}
  Mean ARI over 3 seeds (-- means the method cannot run on that data):
                    single complete  average   kmeans kmedoids gmm_full gmm_diag spectral bernoull
  book blobs         0.250    0.667    0.638    0.659    0.871    0.954    0.648    0.959       --
  rings              1.000    0.002    0.006    0.002    0.002    0.001    0.001    1.000       --
  spirals            1.000    0.019    0.000    0.002    0.002    0.001    0.001    1.000       --
  iris               0.564    0.642    0.759    0.730    0.730    0.904    0.759    0.745       --
  wine              -0.007    0.577   -0.005    0.897    0.741    0.766    0.915    0.850       --
  digits             0.000    0.370    0.390    0.614    0.618    0.501    0.492    0.776       --
  digits, binary     0.000    0.272    0.512    0.545    0.403    0.478    0.392    0.609    0.531
  DNA reads          0.000    0.782    0.885       --    0.891       --       --       --       --
  Largest seed-to-seed range per dataset: {'book blobs': 0.005, 'rings': 0.001, 'spirals': 0.001, 'iris': 0.0, 'wine': 0.545, 'digits': 0.137, 'digits, binary': 0.143, 'DNA reads': 0.015}

total 124s
```
