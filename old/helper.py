class Line:
    # line = \sum_i a_i x_i + const
    def __init__(self,a,const):
        self.a = a # numpy array
        self.const = const # float
    
    def mult_const(self,z):
        return Line(self.a * z,self.const * z)
    
    def add(self,l):
        return Line(self.a + l.a,self.const + l.const)
    
    def add_const(self,z):
        return Line(self.a,self.c + z)
    
class LinearConstraints:
    def __init__(self, lower, upper, x_bound, y_bound):
        self.lower = lower
        self.upper = upper
        self.bound = bound
        
    def eval_box(self):
        lower_val = self.lower.const + self.lower.a.dot(np.array(self.lower.a < 0,dtype=float)*self.bound.upper + np.array(self.lower.a >= 0,dtype=float)*self.bound.lower)
        
        upper_val = self.upper.const + self.upper.a.dot(np.array(self.upper.a >= 0,dtype=float)*self.bound.upper + np.array(self.upper.a < 0,dtype=float)*self.bound.lower)
        
        return BoxConstraints(lower_val,upper_val)
    
    def eval_constraints(self):
        lower_val = self.lower.c + (self.x_bound.upper if self.lower.a < 0 else self.x_bound.lower) + (self.y_bound.upper if self.lower.b < 0 else self.y_bound.lower)
        upper_val = self.upper.c + (self.x_bound.upper if self.upper.a > 0 else self.x_bound.lower) + (self.y_bound.upper if self.upper.b > 0 else self.y_bound.lower)
        return LinearConstraints(Line2D(0,0,lower_val),Line2D(0,0,upper_val),self.x_bounds,self.y_bounds)
    
    def add(self,l):
        lower_line = self.lower.add(l.lower)
        upper_line = self.upper.add(l.upper)
        return LinearConstraints(lower_line, upper_line, self.x_bound, self.y_bound)
    
    def add_const(self,z):
        lower_line = self.lower.add(Line2D(0,0,z))
        upper_line = self.upper.add(Line2D(0,0,z))
        return LinearConstraints(lower_line, upper_line, self.x_bound, self.y_bound)
    
    def mult_const(self,z):
        if z < 0:
            upper_line = self.lower.mult_const(z)
            lower_line = self.upper.mult_const(z)
        else:
            lower_line = self.lower.mult_const(z)
            upper_line = self.upper.mult_const(z)
        return LinearConstraints(lower_line, upper_line, self.x_bound, self.y_bound)
    
    def mult(self,l2):
        l1 = self
        l1_box = self.eval_box()
        l2_box = l2.eval_box()
        
#         c.append(w >= xU*y + x*yU - xU*yU)
#         c.append(w <= xU*y + x*yL - xU*yL)

        lower = (l2.upper.mult_const(l1_box.upper) if l1_box.upper < 0 else l2.lower.mult_const(l1_box.upper))
        lower = lower.add((l1.upper.mult_const(l2_box.upper) if l2_box.upper < 0 else l1.lower.mult_const(l2_box.upper)))
        lower = lower.add_const(-1*l1_box.upper*l2_box.upper)
        
        upper = (l2.upper.mult_const(l1_box.upper) if l1_box.upper > 0 else l2.lower.mult_const(l1_box.upper))
        upper = upper.add((l1.upper.mult_const(l2_box.lower) if l2_box.lower > 0 else l1.lower.mult_const(l2_box.lower)))
        upper = upper.add_const(-1*l1_box.upper*l2_box.lower)
        
        return LinearConstraints(lower,upper,self.x_bound,self.y_bound)

class BoxConstraints:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper


