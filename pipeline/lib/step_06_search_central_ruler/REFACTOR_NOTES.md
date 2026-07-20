# Step 06 central-ruler refactor

This build keeps the existing public wrapper/API but changes candidate construction and ranking.

## Implemented

- ROI-relative endpoint bands and reach gaps.
- Unique interval coverage, so overlapping Hough fragments do not multiply coverage.
- Hough-fragment non-maximum suppression with retained `source_line_indices`.
- Directed top-to-bottom path search instead of undirected connected components.
- Linear point-cloud weighting per fragment instead of approximately quadratic length weighting.
- Original evidence only during candidate selection; adjustment, rescue, extension, and axis harmonization are not used for ranking.
- One normalized `final_score` as the authoritative ranking value.
- Geometry score breakdown, hard-validation diagnostics, and rejection reasons.
- Row-wise bidirectional chamfer mirror score over the edge image inside the ROI.
- Candidate diversity/pool limits and final geometric deduplication are now active.
- Additional metadata: NMS counts, geometry/mirror scores, coverage, span, fit residuals, balance, and validation status.

## Verification performed

Synthetic tests passed for:

1. ROI-relative endpoint metrics.
2. Duplicate Hough fragment suppression.
3. Directed chain behavior in a bridged two-group example.
4. Linear fragment influence in final fitting.
5. Final-score-first ranking.
6. Mirror symmetry preferring a known center axis.
7. End-to-end central-axis search on synthetic fragmented evidence.

The uploaded archives did not include the original Step 05 JSON inputs and ROI masks, so the full IMG_0502–IMG_0512 dataset could not be rerun here. Run the existing Step 06 command on those inputs and inspect the new score breakdown before tuning thresholds.
