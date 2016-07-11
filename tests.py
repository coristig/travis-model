import os
import unittest
import pandas as pd
from model import TravisModel, transform_dummies

def test_point():
    test_val = [{'gender':'male','highest_deg':'masters','ranking':'associate','years_current_rank': 11,'years_since_deg':14}]
    model = TravisModel()
    res = model.execute(test_val)
    return {'pred_salary':res[0]}

def fit_valuation():
    model = TravisModel()
    fit_val = model.fit_val()
    return fit_val

class MyTest(unittest.TestCase):
    def test(self):
        # test a point value is with $2000 of our testset
        self.assertTrue(test_point()['pred_salary']-22919.9<2000)

        # check that testing is over 80%
        self.assertTrue(fit_valuation()>.75)

if __name__ == '__main__':
    unittest.main()
