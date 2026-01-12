data_postprocessor = dict(
    # TODO knn classifier!
    type="NMSPostProcessor",
    score_thr=0.01,
    nms_thr=0.65,
    pool_thr=0.9,
)

data_preprocessor = dict(
    type="WildLifeReIDPreprocessor",
)
