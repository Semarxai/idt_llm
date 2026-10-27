import unittest
import numpy as np
from serial_ci import ci_residual,block_ci_test,contribution,encode
class Tests(unittest.TestCase):
    def test_factorized_null(self):
        ps=np.array([.4,.6]);pa=np.array([[.3,.7],[.6,.4]]);py=np.array([[.8,.2],[.1,.9]])
        p=ps[:,None,None]*pa[:,:,None]*py[:,None,:]
        d,p0=ci_residual(p)
        np.testing.assert_allclose(d,0,atol=1e-15);np.testing.assert_allclose(p0,p)
    def test_null_vector_sums(self):
        p=np.random.default_rng(1).dirichlet(np.ones(27)).reshape(3,3,3)
        d,p0=ci_residual(p)
        np.testing.assert_allclose(d.sum(1),0,atol=1e-15)
        np.testing.assert_allclose(d.sum(2),0,atol=1e-15)
        self.assertAlmostEqual(p0.sum(),1)
    def test_reproducible_and_label_invariant(self):
        rng=np.random.default_rng(2);S=rng.integers(2,size=200);A=rng.integers(2,size=200);Y=rng.integers(2,size=200)
        a=block_ci_test(S,A,Y,block_length=8,draws=99,seed=3)
        b=block_ci_test(S,1-A,1-Y,block_length=8,draws=99,seed=3)
        self.assertAlmostEqual(a['statistic'],b['statistic']);self.assertEqual(a['p_block'],b['p_block'])
    def test_reject_invalid(self):
        with self.assertRaises(ValueError):block_ci_test(np.arange(5),np.arange(4),np.arange(5),block_length=2)
    def test_exact_independence_stat_zero(self):
        triples=np.tile(np.array([[s,a,y] for s in range(2) for a in range(2) for y in range(2)]),(20,1))
        out=block_ci_test(*triples.T,block_length=4,draws=99,seed=42)
        self.assertEqual(out['statistic'],0);self.assertEqual(out['p_block'],1)
if __name__=='__main__':unittest.main()
