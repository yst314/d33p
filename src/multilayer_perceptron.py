import numpy as np
from perceptron import LogicPerceptron

class MultilayerPerceptron(object):
    def __init__(self):
        self.p_and = LogicPerceptron()
        self.p_and.set_and_gate()
        self.p_or = LogicPerceptron()
        self.p_or.set_or_gate()
        self.p_nand = LogicPerceptron()
        self.p_nand.set_nand_gate()
    
    def __call__(self, x1, x2):
        y = self.forward(x1, x2)

        return y

    def forward(self, x1, x2):
        return NotImplementedError

class XorGate(MultilayerPerceptron):
    def forward(self, x1, x2):
        y = self.p_and(self.p_nand(x1, x2), self.p_or(x1, x2))
        return y

if __name__ == '__main__':
    xor = XorGate()
    print(xor(0,0))
    print(xor(1,0))
    print(xor(0,1))
    print(xor(1,1))
