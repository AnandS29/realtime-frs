import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import cvxpy

# Fast Implementation

class Line:
    # line = ax + by + c
    # coeffs = {"x":a, "y":b, "_const":c}
    def __init__(self,coeffs):
        self.coeffs = coeffs.copy()
    
    def mult_const(self,z):
        return Line({key:value*z for (key,value) in self.coeffs.items()})
    
    def add(self,l):
        return Line({key:(value + l.coeffs[key]) for (key,value) in self.coeffs.items()})
    
    def add_const(self,z):
        coeffs = self.coeffs.copy()
        coeffs["_const"] = coeffs["_const"] + z
        return Line(coeffs)
    
    def copy(self):
        return Line(self.coeffs.copy())
    
    def eval_line(self, var_bounds):
        min_val, max_val = self.coeffs["_const"], self.coeffs["_const"]
        
        for (key, value) in var_bounds.items():
            min_val += (value.lower if self.coeffs[key] > 0 else value.upper)*self.coeffs[key]
            max_val += (value.lower if self.coeffs[key] < 0 else value.upper)*self.coeffs[key]
        
        return BoxConstraints(min_val, max_val)
    
class LinearConstraints:

    def __init__(self, lower, upper, var_bounds):
        # lower/upper = Line2D object
        self.lower = lower
        self.upper = upper
        
        # var_bounds = {"x":BoxContstraints(-1,1), "y":BoxContstraints(-1,1)}
        self.var_bounds = var_bounds.copy()
        
    def add(self,l):
        lower_line = self.lower.add(l.lower)
        upper_line = self.upper.add(l.upper)
        return LinearConstraints(lower_line, upper_line, self.var_bounds)
    
    def add_const(self,z):
        lower_line = self.lower.add_const(z)
        upper_line = self.upper.add_const(z)
        return LinearConstraints(lower_line, upper_line, self.var_bounds)
    
    def mult_const(self,z):
        if z < 0:
            upper_line = self.lower.mult_const(z)
            lower_line = self.upper.mult_const(z)
        else:
            lower_line = self.lower.mult_const(z)
            upper_line = self.upper.mult_const(z)
        return LinearConstraints(lower_line, upper_line, self.var_bounds)
    
    def eval_box(self):
        lower_val = sum([(value if key == "_const" else (self.var_bounds[key].upper if value < 0 else self.var_bounds[key].lower)*value) for (key,value) in self.lower.coeffs.items()])
        upper_val = sum([(value if key == "_const" else (self.var_bounds[key].upper if value > 0 else self.var_bounds[key].lower)*value) for (key,value) in self.upper.coeffs.items()])
        
        return BoxConstraints(lower_val,upper_val)
    
    def power(self, d, convex=True):
        # Return constraint for self^d (e.g x^d) for d \geq 1
        constraint = None
        
        if d == 0:
                upper_coeff = {key:0 for key in self.var_bounds.keys()}
                lower_coeff = {key:0 for key in self.var_bounds.keys()}
                upper_coeff["_const"] = 1
                lower_coeff["_const"] = 1
                constraint = LinearConstraints(Line(lower_coeff),Line(upper_coeff),self.var_bounds)
        if d == 1:
            return self.copy()
        
        if convex:
            if d % 2 == 0:
                fn = lambda v: v["x^d"]**d
                gradient_fn = lambda v: {"x^d":d*(v["x^d"]**(d-1))}
                x_d = ConvexConstraints({"x^d":self},self.var_bounds, fn, gradient_fn, convex=True)
                constraint = x_d
            else:
                return self.copy().mult(self.power(d-1,convex))
        else:
            for i in range(d):
                if i == 0:
                    constraint = self.copy()
                else:
                    constraint = constraint.mult(self.copy())

            root_power = d - np.floor(d)
            if root_power > 0:
                constraint = constraint.mult(None)
            
        return constraint
    
    def mult(self,l2):
        l1 = self
        l1_box = self.eval_box()
        l2_box = l2.eval_box()
        
        # Form of lower bound: w >= xU*y + x*yU - xU*yU, w >= xL*y + x*yL - xL*yL
        # Form of upper bound: w <= xU*y + x*yL - xU*yL, w <= xL*y + x*yU - xL*yU
        
        get_lower = lambda l: l.eval_line(self.var_bounds).lower
        get_upper = lambda l: l.eval_line(self.var_bounds).upper

        lower_1 = (l2.upper.mult_const(l1_box.upper) if l1_box.upper < 0 else l2.lower.mult_const(l1_box.upper))
        lower_1 = lower_1.add((l1.upper.mult_const(l2_box.upper) if l2_box.upper < 0 else l1.lower.mult_const(l2_box.upper)))
        lower_1 = lower_1.add_const(-1*l1_box.upper*l2_box.upper)
        
        lower_2 = (l2.upper.mult_const(l1_box.lower) if l1_box.lower < 0 else l2.lower.mult_const(l1_box.lower))
        lower_2 = lower_2.add((l1.upper.mult_const(l2_box.lower) if l2_box.lower < 0 else l1.lower.mult_const(l2_box.lower)))
        lower_2 = lower_2.add_const(-1*l1_box.lower*l2_box.lower)
        
        lowers = [lower_1, lower_2]
        
        use_lower = (0 if get_lower(lower_1) > get_lower(lower_2) else 1)
        
        upper_1 = (l2.upper.mult_const(l1_box.upper) if l1_box.upper > 0 else l2.lower.mult_const(l1_box.upper))
        upper_1 = upper_1.add((l1.upper.mult_const(l2_box.lower) if l2_box.lower > 0 else l1.lower.mult_const(l2_box.lower)))
        upper_1 = upper_1.add_const(-1*l1_box.upper*l2_box.lower)
        
        upper_2 = (l2.upper.mult_const(l1_box.lower) if l1_box.lower > 0 else l2.lower.mult_const(l1_box.lower))
        upper_2 = upper_2.add((l1.upper.mult_const(l2_box.upper) if l2_box.upper > 0 else l1.lower.mult_const(l2_box.upper)))
        upper_2 = upper_2.add_const(-1*l1_box.lower*l2_box.upper)
        
        use_upper = (0 if get_upper(upper_1) < get_upper(upper_2) else 1)
        
        uppers = [upper_1, upper_2]
        
        return LinearConstraints(lowers[use_lower],uppers[use_upper],self.var_bounds)
    
    def div(self,l2):
        # TODO
        return None
    
    def exp(self,d):
        # d > 1 corresponds to number of terms in Taylor expansion for e^x
        constraint = None
        for n in range(d):
            if n == 0:
                continue
            coeff = 1/(np.math.factorial(n))
            if n == 1:
                constraint = self.power(n).mult_const(coeff)
            else:
                constraint = constraint.add(self.power(n).mult_const(coeff))
        constraint = constraint.add_const(1)
        return constraint
    
    def sin(self,constant=False,taylor=None,adaptive=False,debug=False):
        # taylor = d > 1 corresponds to n in \sum_{k=0}^n ... (ie the sigma notation for taylor series) = number of terms - 1
        
        constraint = None
        
        if adaptive:
            box = self.eval_box()
            lower, upper = box.lower, box.upper
            
            if debug: print("Bounds: ", lower, upper)
            
            fn = lambda v: np.sin(v["_self"])
            gradient_fn = lambda v: {"_self":np.cos(v["_self"])}
            
            if upper <= np.pi:
                if debug: print("Concave")
                return ConvexConstraints({"_self":self.copy()}, self.var_bounds, fn, gradient_fn, convex=False)
            elif lower >= np.pi:
                if debug: print("Convex")
                return ConvexConstraints({"_self":self.copy()}, self.var_bounds, fn, gradient_fn, convex=True)
            else:
                if debug: print("Neither")
#                 lower_point, upper_point = 4.50, 1.78 # Use numerical solver to find safe solution?
#                 var_bounds = self.var_bounds
                
#                 # bound of the form ax + b
#                 # sin(x_1) - x_1 cos(x_1)
#                 b = lambda t: np.sin(t) - t*np.cos(t)
#                 # cos(x_1)
#                 a = lambda t: np.cos(t)
#                 lower_bound = self.copy().mult_const(a(lower_point)).add_const(b(lower_point)).lower
#                 upper_bound = self.copy().mult_const(a(upper_point)).add_const(b(upper_point)).upper
#                 return LinearConstraints(lower_bound, upper_bound, var_bounds)
                constant = True;
            
#             box = self.eval_box()
#             xL,xU = box.lower, box.upper
#             x_avg = 0.5*(xL+xU)

#             m = (np.sin(xU) - np.sin(xL))/(xU - xL)

#             if xU <= np.pi:
#                 # Concave
#                 ln_upper = self.mult_const(np.cos(x_avg)).add_const(np.sin(x_avg)-np.cos(x_avg)*x_avg).upper
#                 ln_lower = self.mult_const(m).add_const(np.sin(xL) - xL*m).lower
#                 return LinearConstraints(ln_lower, ln_upper, self.var_bounds)

#             elif xL >= np.pi:
#                 # Convex
#                 ln_upper = self.mult_const(np.cos(x_avg)).add_const(np.sin(x_avg)-np.cos(x_avg)*x_avg).lower
#                 ln_lower = self.mult_const(m).add_const(np.sin(xL) - xL*m).upper
#                 return LinearConstraints(ln_lower, ln_upper, self.var_bounds)

#             else:
#                 constant = True
                
        if taylor is not None and not constant:
            d = taylor
            constraint = None
            assert d > 1, "d must be greater than 1"
            for k in range(d):
                coeff = ((-1)**k)/(np.math.factorial(2*k + 1))
                power = 2*k + 1
                if k == 0:
                    constraint = self.power(power).mult_const(coeff)
                else:
                    constraint = constraint.add(self.power(power).mult_const(coeff))
        
        if constant:
            upper_coeff = {key:0 for key in self.var_bounds.keys()}
            lower_coeff = {key:0 for key in self.var_bounds.keys()}
            upper_coeff["_const"] = 1
            lower_coeff["_const"] = -1
            constraint = LinearConstraints(Line(lower_coeff),Line(upper_coeff),self.var_bounds)
        return constraint
    
    def cos(self,constant=False,taylor=None,adaptive=False, debug=False):
        # taylor = d > 1 corresponds to n+1 in \sum_{k=0}^n ... (ie the sigma notation for taylor series) = number of terms - 1
        
        if adaptive:
            box = self.eval_box()
            lower, upper = box.lower, box.upper
            
            if debug: print("Bounds: ", lower, upper)
            
            fn = lambda v: np.cos(v["_self"])
            gradient_fn = lambda v: {"_self":-1*np.sin(v["_self"])}
            
            if (upper <= np.pi/2 or lower >= 1.5*np.pi):
                if debug: print("Concave")
                return ConvexConstraints({"_self":self.copy()}, self.var_bounds, fn, gradient_fn, convex=False)
            elif (lower >= np.pi/2 and upper <= 1.5*np.pi):
                if debug: print("Convex")
                return ConvexConstraints({"_self":self.copy()}, self.var_bounds, fn, gradient_fn, convex=True)
            else:
                if debug: print("Neither")
                constant = True

        if taylor is not None and not constant:
            d = taylor
            assert d > 1, "d must be greater than 1"
            constraint = None
            for k in range(d):
                coeff = ((-1)**k)/(np.math.factorial(2*k))
                power = 2*k
                # Change power function to accept d = 0 (ie x^0 = 1), so we get bounds 1 <= x^0 <= 1
                if k == 0:
                    constraint = self.power(power).mult_const(coeff)
                else:
                    constraint = constraint.add(self.power(power).mult_const(coeff))
        
        if constant:
            upper_coeff = {key:0 for key in self.var_bounds.keys()}
            lower_coeff = {key:0 for key in self.var_bounds.keys()}
            upper_coeff["_const"] = 1
            lower_coeff["_const"] = -1
            constraint = LinearConstraints(Line(lower_coeff),Line(upper_coeff),self.var_bounds)
        return constraint
    
    def __mul__(self, other):
        if not isinstance(other,LinearConstraints):
            return self.mult_const(other)
        return self.mult(other)
    
    def __add__(self, other):
        if not isinstance(other,LinearConstraints):
            return self.add_const(other)
        return self.add(other)
    
    def __sub__(self, other):
        if not isinstance(other,LinearConstraints):
            return self.add_const(-1*other)
        return self.add(other.mult_const(-1))
    
    def __rmul__(self, other):
        if not isinstance(other,LinearConstraints):
            return self.mult_const(other)
        return self.mult(other)
    
    def __radd__(self, other):
        if not isinstance(other,LinearConstraints):
            return self.add_const(other)
        return self.add(other)
    
    def __rsub__(self, other):
        if not isinstance(other,LinearConstraints):
            return self.add_const(-1*other)
        return self.add(other.mult_const(-1))
    
    def copy(self):
        return LinearConstraints(self.lower.copy(),self.upper.copy(),self.var_bounds.copy())

class ConvexConstraints(LinearConstraints):
    def __init__(self, input_variables, var_bounds, fn, gradient_fn, convex=True, point_grad=None):
        # Convex = False \implies Concave
        self.convex = convex
        
        # var_bounds = {"x":LinearConstraint(), "y":LinearConstraint()}
        self.variables = input_variables.copy()
        
        self.var_bounds = var_bounds
        
        # Point around which to take gradient
        if point_grad is None:
            # Take gradient in the middle of bounds
            self.point = {key:(0.5*(value.eval_box().upper + value.eval_box().lower)) for (key,value) in self.variables.items()}
        else:
            self.point = point_grad
        
        # {"x":1, "y":1} |-> x \in \mathbb{R} (evaluate fn based on dictioary input)
        self.func = fn
        
        # {"x":1, "y":1} |-> {"x":df/dx({"x":1, "y":1}), "y":df/dy({"x":1, "y":1})} (key corresponds to partial x at value from input)
        self.grad = gradient_fn
        
        # Find lower and upper bounds
        lower, upper = self.compute_bounds()
        
        # lower/upper = Line2D object
        self.lower = lower
        self.upper = upper
        
        super().__init__(self.lower,self.upper,self.var_bounds)
        
    def compute_bounds(self):
        ## TODO: Double check this bound.
        
        ## Find bound using convexity
        # f(x) - \sum_{v \in {x,y}} x_v * df/dv(x)
        grad = self.grad(self.point)
        const = self.func(self.point) - sum([self.point[key]*value for (key,value) in grad.items()])
        bound_convexity = None
        for key in self.variables.keys():
            var = self.variables[key]
            coeff = grad[key]
            if bound_convexity is None:
                bound_convexity = var.mult_const(coeff)
            else:
                bound_convexity = bound_convexity.add(var.mult_const(coeff))
            
        bound_convexity = bound_convexity.add_const(const)
        
        ## Find bounds using endpoints
        keys = [key for (key,value) in self.variables.items()]
        v_bounds = [[self.variables[key].eval_box().lower, self.variables[key].eval_box().upper] for key in keys]
        A = list(itertools.product(*v_bounds))
        
        # Add fn values
        z = [self.func({keys[i]:a[i] for i in range(len(keys))}) for a in A]
        A_temp = []
        for i in range(len(A)):
            a = A[i]
            a_temp = list(a)
            a_temp.append(z[i])
            A_temp.append(a_temp)
        A = np.array(A_temp)
        
        # n \cdot <x-x_0, y-y_0> = 0
        b = np.ones((A.shape[0],1))
        n = np.matmul(np.linalg.pinv(A),b)
        
        # Find in terms of f(x): a_N f(x) + \sum_i a_i x_i = b_i        
        bound_plane = None
        a_N = n[len(keys)].tolist()[0]
        const = 1/a_N
        for i in range(len(keys)):
            key = keys[i]
            var = self.variables[key]
            coeff = (-1*n[i]/a_N).tolist()[0]

            if bound_plane is None:
                bound_plane = var.mult_const(coeff)
            else:
                bound_plane = bound_plane.add(var.mult_const(coeff))
        bound_plane = bound_plane.add_const(const)

        if self.convex:
            lower, upper = bound_convexity.lower, bound_plane.upper
        else:
            lower, upper = bound_plane.lower, bound_convexity.upper
        
        return lower, upper
    
class BoxConstraints:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

def make_variables(var_bounds):
    constraints = {}
    for key in var_bounds.keys():
        coeffs = {k:0 for k in var_bounds.keys()}
        coeffs["_const"] = 0
        coeffs[key] = 1
        constraints[key] = LinearConstraints(Line(coeffs.copy()), Line(coeffs.copy()),var_bounds)
    return constraints

def linear_to_polytope(var_bounds,variables,order):
    keys = list(var_bounds.keys())
    A = []
    b = []
    for i in range(len(keys)):
        var = var_bounds[keys[i]]
        A.append([(1 if i == j else 0) for j in range(len(keys) + len(order))])
        b.append(var.upper)
        A.append([(-1 if i == j else 0) for j in range(len(keys) + len(order))])
        b.append(-1*var.lower)
    
    for var_name in order:
        variable = variables[var_name]
        lower, upper = variable.lower.coeffs, variable.upper.coeffs
        row = []
        for key in keys:
            row.append(lower[key])
        for var in order:
            row.append((-1 if var == var_name else 0))
        A.append(row)
        b.append(-1*lower["_const"])
        
        row = []
        for key in keys:
            row.append(-1*upper[key])
        for var in order:
            row.append((1 if var == var_name else 0))
        A.append(row)
        b.append(upper["_const"])
    
    A = np.array(A)
    b = np.array([b]).T
    proj_dim = [i for i in range(len(keys) + len(order)) if i >= len(keys)]
    return A, b, proj_dim

# CVX-based Methods

def McCormick(w,x,y,x_bounds,y_bounds):
    xL,xU = x_bounds
    yL,yU = y_bounds

    c = []
#     c.append(w >= xL*y + x*yL - xL*yL)
    c.append(w >= xU*y + x*yU - xU*yU)
    c.append(w <= xU*y + x*yL - xU*yL)
#     c.append(w <= x*yU + xL*y - xL*yU)
    return c

def bounds(x,constraints,Npast=0):

    objective_max = cvxpy.Maximize(x)
    problem_maximum = cvxpy.Problem(objective_max,constraints[-Npast:])
    value_max = problem_maximum.solve()

    objective_min = cvxpy.Minimize(x)
    problem_minimum = cvxpy.Problem(objective_min,constraints[-Npast:])
    value_min = problem_minimum.solve() #solver=cvxpy.GUROBI

    return (value_min,value_max)

def sin_adaptive_cvx(w,x,x_bounds,debug=False):
    xL,xU = x_bounds
    x_avg = 0.5*(xL+xU)
    c = []
    
    m = (np.sin(xU) - np.sin(xL))/(xU - xL)
#     print(xL,xU,m,x_avg)
    
    if xU <= np.pi:
        # Concave
        c.append(w <= np.sin(x_avg) + np.cos(x_avg)*(x - x_avg))
        c.append(w >= (x-xL)*m + np.sin(xL))
        
        if debug:
            coeffs_lower = {"_const":np.sin(xL)-m*xL, "x_sin":m}
            coeffs_upper = {"_const":np.sin(x_avg)-np.cos(x_avg)*x_avg, "x_sin":np.cos(x_avg)}
            print("sin(x) lower: ", coeffs_lower)
            print("sin(x) upper: ", coeffs_upper)
        
    elif xL >= np.pi:
        # Convex
        c.append(w >= np.sin(x_avg) + np.cos(x_avg)*(x - x_avg))
        c.append(w <= (x-xL)*m + np.sin(xL))
        
        if debug:
            coeffs_upper = {"_const":np.sin(xL)-m*xL, "x_sin":m}
            coeffs_lower = {"_const":np.sin(x_avg)-np.cos(x_avg)*x_avg, "x_sin":np.cos(x_avg)}
            print("sin(x) lower: ", coeffs_lower)
            print("sin(x) upper: ", coeffs_upper)
        
    else:
#         lower_point, upper_point = 4.50, 1.78 # Use numerical solver to find safe solution?
#         # sin(x_1) - x_1 cos(x_1)
#         b = lambda t: np.sin(t) - t*np.cos(t)
#         # cos(x_1)
#         a = lambda t: np.cos(t)
#         c.append(w >= a(lower_point)*x + b(lower_point))
#         c.append(w <= a(upper_point)*x + b(upper_point))
        c.append(w >= -1)
        c.append(w <= 1)
        
#         if debug:
#             coeffs_lower = {"_const":b(lower_point), "x_sin":a(lower_point)}
#             coeffs_upper = {"_const":b(upper_point), "x_sin":a(upper_point)}
#             print("sin(x) lower: ", coeffs_lower)
#             print("sin(x) upper: ", coeffs_upper)
        
    return c

def cos_adaptive_cvx(w,x,x_bounds,debug=False):
    xL,xU = x_bounds
    x_avg = 0.5*(xL+xU)
    c = []
    
    m = (np.cos(xU) - np.cos(xL))/(xU - xL)
#     print(xL,xU,m,x_avg)

    
    if (xU <= np.pi/2 or xL >= 1.5*np.pi):
        # Concave
        c.append(w <= np.cos(x_avg) - np.sin(x_avg)*(x - x_avg))
        c.append(w >= (x-xL)*m + np.cos(xL))
        
        if debug:
            coeffs_lower = {"_const":np.cos(xL)-m*xL, "x_cos":m}
            coeffs_upper = {"_const":np.cos(x_avg)+np.sin(x_avg)*x_avg, "x_cos":-1*np.sin(x_avg)}
            print("cos(x) lower: ", coeffs_lower)
            print("cos(x) upper: ", coeffs_upper)
        
    elif (xL >= np.pi/2 and xU <= 1.5*np.pi):
        # Convex
        c.append(w >= np.cos(x_avg) - np.sin(x_avg)*(x - x_avg))
        c.append(w <= (x-xL)*m + np.cos(xL))
        
        if debug:
            coeffs_upper = {"_const":np.cos(xL)-m*xL, "x_cos":m}
            coeffs_lower = {"_const":np.cos(x_avg)+np.sin(x_avg)*x_avg, "x_cos":-1*np.sin(x_avg)}
            print("cos(x) lower: ", coeffs_lower)
            print("cos(x) upper: ", coeffs_upper)
        
    else:
        # Constant
        c.append(w >= -1)
        c.append(w <= 1)
        
        if debug:
            coeffs_lower = {"_const":-1, "x_cos":0}
            coeffs_upper = {"_const":1, "x_cos":0}
            print("cos(x) lower: ", coeffs_lower)
            print("cos(x) upper: ", coeffs_upper)
        
    return c