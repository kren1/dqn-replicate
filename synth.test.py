#!/usr/bin/python3
import unittest
from synth import NNExpr, Harness

class TestNNExpr(unittest.TestCase):
  def test_onePlusOne(self):
    e = NNExpr("1+1")
    self.assertEqual(e.evalExpr(), 2)
  def test_complexExpr(self):
    e = NNExpr("1+2+(3-4)+5-6+7+8-9")
    self.assertEqual(e.evalExpr(), 7)
  def test_syntaxError(self):
    e = NNExpr("1+(-1")
    with self.assertRaises(SyntaxError):
      e.evalExpr()
  def test_idompotence(self):
    e = NNExpr("1+2+(3-4)+5-6+7+8-9")
    self.assertEqual(e.evalExpr(), NNExpr(str(e)).evalExpr())

class TestHarness(unittest.TestCase):
  def test_simple(self):
    h = Harness(NNExpr("1+1"))
    self.assertEqual(h.act(2, False), 1)



if __name__ == '__main__':
  unittest.main(verbosity=2)
