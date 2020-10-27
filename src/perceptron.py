import numpy as np

class LogicPerceptron(object):
    def __init__(self, w, bias):
        self.w = w
        self.bias = bias

    def __call__(self, x1, x2, debug=False):
        x = np.array([x1, x2])
        y = np.sum(self.w*x) + self.bias

        if debug:
            print(f"x weighted:{x[0]}")
            print(f"y weighted:{x[1]}")
            print(f"output    :{y}\n")

        if y <= 0:
            return 0
        else:
            return 1

    def test(self, debug=False):
        return (f"{self(0, 0, debug)}"
                f"{self(1, 0, debug)}"
                f"{self(0, 1, debug)}"
                f"{self(1, 1, debug)}")
    
class AndGate(LogicPerceptron):
    def __init__(self):
        w = np.array([0.5, 0.5])
        super(LogicPerceptron, self).__init__(np.array([0.5, 0.5]), -0.7)

class NandGate(LogicPerceptron):
    def __init__(self):
        super.__init__(np.array([-0.5, -0.5]), 0.7)

class OrGate(LogicPerceptron):
    def __init__(self):
        super.__init__(np.array([0.5, 0.5]), -0.2)
    

if __name__ == '__main__':
    p_and = AndGate()
    p_or = OrGate()
    p_nand = NandGate()

    #p.debug_calc_return()
    ret = p_and.test()

    print("x:0101")
    print("y:0011")
    print(f"r:{ret}")
