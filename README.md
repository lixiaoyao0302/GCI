# GCI

Core C++ source code for point-cloud registration. Requires C++17, PCL and Eigen.

## Build

Install PCL and its dependencies, then run from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If PCL is not found, pass `-DPCL_DIR=/path/to/pcl/cmake` to the first command. On Windows, use a Visual Studio developer shell. OpenMP is optional. This builds the `gci_core` library; no command-line executable is included. The core was compiled locally with MSVC and PCL 1.13.0; other environments have not been verified.

## Input interface

Include `GCI.h` and link your application to `gci_core`. Supply four `pcl::PointCloud<pcl::PointXYZ>` pointers and an explicitly configured `gci::Options` object:

```cpp
// All cloud pointers and options below are supplied by the calling application.
const auto result = gci::registerPointClouds(
    source_reference, target_reference,
    source_correspondences, target_correspondences, options);
```

- Reference clouds describe the source and target surfaces and must be nonempty.
- Correspondence clouds must have equal sizes, with at least three matches; source row `i` matches target row `i`.
- Coordinates must be finite. Each correspondence cloud uses its respective reference cloud's coordinate system. All coordinates and length parameters use the same units.
- Ground truth is not an input. Feature extraction and correspondence generation are supplied by the calling application.

Set the options declared in `include/GCI.h` for your data. They control sampling budgets, graph construction, noise scales, posterior inference, observation selection, pose refinement, component switches and threading. Zero initialization is not a usable parameter preset. Use positive noise scales, regularization and convergence tolerances; an inlier prior in `(0,1)` and damping in `[0,1)`; nonzero iteration/hypothesis budgets; and correspondence/selection budgets and normal-neighbor counts of at least three. Set all four component switches to `true` for the full method; `false` selects an ablation. Set `tangent_normal_ratio > 1`. `relation_threshold` is the edge threshold (tau_g), whereas `relation_sigma` is the Gaussian scale (sigma_g). `mean_field_damping` retains the old probability, so the paper coefficient gamma equals `1 - mean_field_damping`. Optional `correspondence_priors` use the original match order; an empty vector uses `prior_inlier`. `hypothesis_count` caps the total tested candidates. `edge_degree_budget=0` retains all compatible edges as described in the method; positive values enable optional graph pruning. No paper experiment preset is provided.

## Output

`result.transformation` is a 4×4 matrix mapping source points into the target coordinate system. Check `result.success` and `result.message` before using it. Execution success does not establish registration accuracy or convergence; inspect `result.diagnostics` and evaluate accuracy separately. `gci::writeDiagnosticsJson(path, result)` saves diagnostics to a caller-supplied path. Information diagnostics use the summed posterior-weighted data matrix without the selection regularizer. Rank-deficient covariance traces and condition numbers are infinite in the C++ result and `null` in JSON; the reported ranks distinguish these cases. `converged` requires the outer rotation, translation and objective stopping criteria together.

## Datasets and acknowledgements

The WHU-TLS and ETH datasets used in this study are publicly available from their respective providers:

- [WHU-TLS dataset](https://3s.whu.edu.cn/ybs/en/benchmark.htm)
- [ETH dataset](https://prs.igp.ethz.ch/research/completed_projects/automatic_registration_of_point_clouds.html)

We thank Zhen Dong and all co-authors of the WHU-TLS review and benchmark, and Pascal Willy Theiler, Jan Dirk Wegner, and Konrad Schindler, for their contributions and for making these datasets available to the research community. Please cite the corresponding papers when using the datasets:

- Dong et al. (2020). Registration of Large-Scale Terrestrial Laser Scanner Point Clouds: A Review and Benchmark. *ISPRS Journal of Photogrammetry and Remote Sensing*, 163, 327–342. [DOI](https://doi.org/10.1016/j.isprsjprs.2020.03.013).
- Theiler, Wegner, and Schindler (2014). Keypoint-based 4-points congruent sets—Automated marker-less registration of laser scans. *ISPRS Journal of Photogrammetry and Remote Sensing*, 96, 149–163. [DOI](https://doi.org/10.1016/j.isprsjprs.2014.06.015).

BibTeX entries are provided in [references.bib](references.bib).

