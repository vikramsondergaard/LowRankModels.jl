# Universal Fair Data Projection

Universal Fair Data Projection is a Julia package for applying various notions of fairness to unsupervised learning, projecting data to a fair space for use in fair classification tasks.

To run a test, use the following command:
```bash
julia --project=. test/fair_glrms/experiments/test_<FAIRNESS>.jl adult <INDEPENDENCE_MEASURE> -k <K> -s <S>
```
where `<FAIRNESS>` is a choice between `independence`, `separation`, and `sufficiency`, `<INDEPENDENCE_MEASURE>` is `hsic` (for HSIC), `orthog` (for the hard Pearson correlation constraint), or `softorthog` (for the soft Pearson correlation constraint), `K` is the number of archetypal features with which to perform the dimensionality reduction (2 to 4 is ideal), and `S` is the scale of the fairness trade-off parameter (60.0 is a good start).

Unfortunately, only the Adult data set is available here, since the Australian Ad Observatory data set is proprietary.# Universal Fair Data Projection