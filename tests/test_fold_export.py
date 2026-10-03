import numpy as np
from ssm_decode.fold_export import fold_raw_projection,quantize_columns

def test_fold_matches_normalized_projection_and_quantizes_columns():
 rng=np.random.RandomState(3);raw=rng.poisson(2,(50,5)).astype(np.float32);mean=rng.randn(5).astype(np.float32);std=(rng.rand(5)+.2).astype(np.float32);w=rng.randn(5,7).astype(np.float32)
 p,b=fold_raw_projection(w,mean,std);assert np.max(np.abs(((raw-mean)/std)@w-(raw@p+b)))<1e-5
 q,s=quantize_columns(p);assert q.dtype==np.int8 and s.shape==(7,)
