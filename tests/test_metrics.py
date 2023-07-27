import torch
from torch.testing import assert_close
import pytest

from stepback.metrics import Loss

#==========================================================
## Squared loss
#==========================================================

x1 = torch.arange(10).float()
x2 = torch.flip(x1, dims=(0,))
x1.requires_grad_(True)


def test_mse_loss():

    loss_fn = Loss(name='squared', backwards=True)
    L = loss_fn.compute(x1, x2)

    assert_close(L, torch.tensor(33.))
    assert_close(x1.grad, (2/len(x1))*(x1-x2))

    return


#==========================================================
## Cross entropy
#==========================================================
@pytest.fixture
def ce_tensors():
    classes = torch.tensor([4, 8, 9, 6, 4, 5, 5, 6, 5, 7, 5, 7, 5, 4, 2]).long()
    scores =  torch.tensor([[-0.1959, -0.7762,  0.1785, -1.4649,  1.2669, -0.0604,  0.1576,  0.3380,
            -0.8693,  1.1948],
            [ 1.3242, -0.9562, -0.7279,  1.3285, -1.2064,  0.2183,  2.0099, -0.3865,
            0.7806,  1.1125],
            [ 0.0787,  1.6893, -0.9566,  0.1654, -0.1997,  0.9043,  1.6809,  0.7313,
            0.2039, -0.0813],
            [-0.6521,  1.2596,  0.9010,  0.6886,  0.6985, -0.7931, -0.9661, -1.3514,
            1.7910,  0.6084],
            [-0.0455, -0.3992, -0.2353, -0.5687,  1.1628, -0.8393, -0.0141, -0.8454,
            -1.5971,  0.8566],
            [ 0.8206, -0.7643, -0.5596,  0.4805, -1.3581, -0.1660, -1.3277,  0.5665,
            -1.3233,  1.6270],
            [-0.2347, -0.2372, -0.0619, -1.2430,  1.0813,  0.7433,  0.0546, -0.9523,
            0.9558, -2.3810],
            [ 1.4109, -1.6277, -0.0723, -0.6083,  0.6392, -1.9571,  0.2974,  0.7995,
            0.8891, -0.7055],
            [ 0.4434,  0.8036,  0.0746, -0.1612, -0.3719,  1.8079,  0.0178,  0.2440,
            -1.4762,  0.8238],
            [ 0.7077,  0.1258,  0.3596, -1.1994, -0.5285,  0.1475,  2.0981,  2.1802,
            0.4325, -0.3512],
            [ 1.3532, -0.7847, -0.2228, -0.0164, -1.0240, -0.0640, -0.4821, -0.7863,
            -0.5992, -0.3147],
            [ 0.0651, -1.4360, -1.4678, -0.2500, -1.1704,  0.4134,  2.1859,  0.5841,
            0.9491,  0.1629],
            [ 0.4781, -1.2158, -0.0595, -1.0502,  0.5107,  2.4388,  0.0928,  0.5852,
            -0.1806, -0.1268],
            [-0.5669,  0.4501, -1.2003,  1.3840, -0.4881,  0.2045,  2.3728, -0.1711,
            -1.2818,  1.2713],
            [ 0.4552, -1.1390, -0.4990,  0.4966,  0.4489, -1.5235, -0.1297, -0.0790,
            0.5328,  1.0348]], requires_grad=True)

    grad = torch.tensor([[ 0.0041,  0.0023,  0.0059,  0.0011, -0.0491,  0.0047,  0.0058,  0.0069,
            0.0021,  0.0163],
            [ 0.0108,  0.0011,  0.0014,  0.0108,  0.0009,  0.0036,  0.0213,  0.0019,
            -0.0604,  0.0087],
            [ 0.0034,  0.0172,  0.0012,  0.0038,  0.0026,  0.0079,  0.0171,  0.0066,
            0.0039, -0.0637],
            [ 0.0018,  0.0121,  0.0084,  0.0068,  0.0069,  0.0016, -0.0654,  0.0009,
            0.0206,  0.0063],
            [ 0.0060,  0.0042,  0.0050,  0.0036, -0.0465,  0.0027,  0.0062,  0.0027,
            0.0013,  0.0148],
            [ 0.0113,  0.0023,  0.0028,  0.0080,  0.0013, -0.0625,  0.0013,  0.0088,
            0.0013,  0.0253],
            [ 0.0044,  0.0044,  0.0052,  0.0016,  0.0164, -0.0550,  0.0059,  0.0021,
            0.0145,  0.0005],
            [ 0.0191,  0.0009,  0.0043,  0.0025,  0.0088,  0.0007, -0.0604,  0.0104,
            0.0113,  0.0023],
            [ 0.0060,  0.0086,  0.0041,  0.0033,  0.0027, -0.0432,  0.0039,  0.0049,
            0.0009,  0.0088],
            [ 0.0052,  0.0029,  0.0037,  0.0008,  0.0015,  0.0030,  0.0210, -0.0439,
            0.0040,  0.0018],
            [ 0.0264,  0.0031,  0.0055,  0.0067,  0.0025, -0.0603,  0.0042,  0.0031,
            0.0038,  0.0050],
            [ 0.0038,  0.0009,  0.0008,  0.0028,  0.0011,  0.0054,  0.0319, -0.0602,
            0.0093,  0.0042],
            [ 0.0051,  0.0009,  0.0030,  0.0011,  0.0053, -0.0302,  0.0035,  0.0057,
            0.0027,  0.0028],
            [ 0.0016,  0.0044,  0.0008,  0.0112, -0.0649,  0.0035,  0.0302,  0.0024,
            0.0008,  0.0100],
            [ 0.0086,  0.0017, -0.0634,  0.0089,  0.0085,  0.0012,  0.0048,  0.0050,
            0.0093,  0.0153]])
    return classes, scores, grad



def test_cross_entropy_loss(ce_tensors):
    classes, scores, grad = ce_tensors
    loss_fn = Loss(name='cross_entropy', backwards=True)
    L = loss_fn.compute(scores, classes)

    assert_close(L, torch.tensor(2.1921), atol=1e-4, rtol=1e-4) 
    assert_close(scores.grad, grad, atol=1e-4, rtol=1e-4)

    return

def test_cross_entropy_accuracy(ce_tensors):
    classes, scores, grad = ce_tensors
    loss_fn = Loss(name='cross_entropy_accuracy')
    L = loss_fn.compute(scores, classes)
    assert_close(L, torch.tensor(0.3333), atol=1e-4, rtol=1e-4)

    return

def test_ce_loss_redundancy(ce_tensors):
    """CE loss with redundant dimensions in classes """
    classes, scores, grad = ce_tensors
    loss_fn = Loss(name='cross_entropy', backwards=True)
    L = loss_fn.compute(scores, classes[:,None])

    assert_close(L, torch.tensor(2.1921), atol=1e-4, rtol=1e-4) 
    assert_close(scores.grad, grad, atol=1e-4, rtol=1e-4)

    return

#==========================================================
## Logistic loss
#==========================================================
@pytest.fixture
def logistic_tensors():
    classes = torch.tensor([-1, -1, -1,  1, -1, -1,  1,  1,  1,  1]).long()
    scores = torch.tensor([-1.9476, -1.3508,  0.8898, -0.7867,  0.2676,  0.2802,  0.0994,  1.5020,
                        0.8969, -0.0449], requires_grad=True)

    grad = torch.tensor([ 0.0125,  0.0206,  0.0709, -0.0687,  0.0567,  0.0570, -0.0475, -0.0182,
                    -0.0290, -0.0511])

    return classes, scores, grad

def test_logistic_loss(logistic_tensors):
    classes, scores, grad = logistic_tensors

    loss_fn = Loss(name='logistic', backwards=True)
    L = loss_fn.compute(scores, classes)

    assert_close(L, torch.tensor(0.6342), atol=1e-4, rtol=1e-4) 
    assert_close(scores.grad, grad, atol=1e-4, rtol=1e-4)

    return

def test_logistic_accuracy(logistic_tensors):
    classes, scores, grad = logistic_tensors
    loss_fn = Loss(name='logistic_accuracy')
    L = loss_fn.compute(scores, classes)
    assert_close(L, torch.tensor(0.5), atol=1e-4, rtol=1e-4)

    return

def test_logistic_loss_redundancy(logistic_tensors):
    """Logistic loss with redundant dimensions in classes """
    classes, scores, grad = logistic_tensors
    loss_fn = Loss(name='logistic', backwards=True)
    L = loss_fn.compute(scores[:,None], classes[:,None])

    assert_close(L, torch.tensor(0.6342), atol=1e-4, rtol=1e-4) 
    assert_close(scores.grad, grad, atol=1e-4, rtol=1e-4)

    return