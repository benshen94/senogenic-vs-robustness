# Sibling mortality coordinates

Figure 2d uses digitized plotted summary points from Figure 1, upper
(brothers) panel, of Gavrilova and Gavrilov, *Protective Effects of Familial
Longevity Decrease With Age and Become Negligible for Centenarians*,
The Journals of Gerontology: Series A 77, 736--743 (2022).
The [primary publisher page and Figure 1 caption](https://academic.oup.com/biomedgerontology/article/77/4/736/6470934)
identify brothers of centenarians and brothers of individuals who died at
age 65, plotted on a base-10 logarithmic mortality scale.

The coordinates are in
[`results/tables/fig2d_raw_digitized_points.csv`](../../results/tables/fig2d_raw_digitized_points.csv).
They are digitized summaries, not individual survival records. The
[`Figure 2 renderer`](../../analysis/figures/figure2/plot_fig2_fp.py)
loads this table, selects rows marked `plotted` and the `brothers_short` and
`brothers_cent` series, and fits lines to log10 mortality versus age.
Shaded intervals are 95% OLS mean-fit intervals; they do not include
digitization uncertainty. The supplied coordinates do not record the
digitization date, operator, software, or image version.
