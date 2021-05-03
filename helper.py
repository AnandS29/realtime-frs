import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import cvxpy

# Reachable Set Verification via Sampling

def random_trajs(ranges,forwardEuler,constants,n=10):
    trajs = []
    for i in range(n):
        init = []
        for r in ranges:
            a = np.random.random_sample()
            x0 = r[0] + a*(r[1]-r[0])
            init.append(x0)
        traj = forwardEuler(init,constants,plot=False)
        trajs.append(traj)
    return trajs

def get_final_points(ranges,forwardEuler,constants,n=10,t=-1):
    points = []
    trajs = random_trajs(ranges,forwardEuler,constants,n=n)
    for traj in trajs:
        points.append(traj[t])
    return points

def verify_valid_boxes(boxes,ranges,forwardEuler,constants,n=10,compare=True, debug=False):
    valid = []
    trajs = random_trajs(ranges,forwardEuler=forwardEuler,constants=constants,n=n)
#     print(len(trajs))
    ind = 0
    ind_stop = 0
    for traj in trajs:
        # NOTE: Skipping last part of traj as box has not been added yet
        for i in range(len(traj)-1):
            for j in range(len(traj[i])):
                el = traj[i][j]
                bound = boxes[i][j]
                if el <= bound[1] and el >= bound[0]:
                    valid.append(True)
                else:
                    valid.append(False)
                    ind_stop = ind
                    print(el, bound[0], bound[1])
        ind += 1
                    
    
    # Method of checking traj for where the "problem" is
    if debug:
#         print("Number Valid:", sum(valid), all(valid), any(valid))
        t = trajs[ind_stop]
        for i in range(len(t)-1):
            s = t[i]
            box = boxes[i]
            stop = None
            j = 0
            for b in box:
                if s[j] <= b[0] or s[j] >= b[1]:
                    stop = j
                    break
                j += 1
            if stop is not None:
                print("Time: ", i)
                print(s)
                print(box)
                print(stop)
                print()
     
    if compare:
        # For each state, look at last timestep to find min and max bounds for that state from the sample trajectories
        trajs = np.array(trajs)
        over_approx = []
        # again, ignoring last timestep of traj
        for j in range(trajs.shape[1]-1):
            area = 1
            for i in range(trajs.shape[2]):
                min_bound = np.min(trajs[:,j,i])
                max_bound = np.max(trajs[:,j,i])
                area *= max_bound-min_bound
            over_approx.append(area)
            
        under_approx = []
        for box in boxes:
            area = 1
            for bounds in box:
                area *= bounds[1]-bounds[0]
            under_approx.append(area)
        plt.figure()
        plt.plot(over_approx,label="(Pseudo) Over Approx")
        plt.plot(under_approx,label="Under Approx")
        plt.xlabel("Iteration")
        plt.ylabel("Area")
        plt.title("Area of reachable set for (pseudo) over and under approx")
        plt.legend()
        
    return all(valid)

# Visualizations

def visualize(convex_res,fast_res):
    plt.figure()
    plt.plot(fast_res["area"],label="Fast")
    plt.plot(convex_res["area"],label="CVX")
    plt.xlabel("Iteration")
    plt.ylabel("Area")
    plt.title("Area of reachable set")
    plt.legend()
    
    plt.figure()
    plt.plot(fast_res["time"],label="Fast")
    plt.plot(convex_res["time"],label="CVX")
    plt.xlabel("Iteration")
    plt.ylabel("Time for iteration")
    plt.title("Time taken")
    plt.legend()
    
    plt.figure()
    plt.plot(np.cumsum(fast_res["time"]),label="Fast")
    plt.plot(np.cumsum(convex_res["time"]),label="CVX")
    plt.xlabel("Iteration")
    plt.ylabel("Time for iteration")
    plt.title("Cumulative Time taken")
    plt.legend()
    
    plt.figure()
    plt.plot([convex_res["time"][i]/fast_res["time"][i] for i in range(len(fast_res["time"]))])
    plt.xlabel("Iteration")
    plt.ylabel("Ratio of times")
    plt.title("Ratio of CVX/Fast for time per iteration")
    
def compare_clouds(polytope, random):
    plt.figure()
    for point in polytope:
        plt.scatter(point[0], point[1], c="tab:blue", label="polytope", alpha=0.3)
    for point in random:
        plt.scatter(point[0], point[1], c="tab:orange", label="random", alpha=0.3)
    
    plt.xlabel("X")
    plt.ylabel("Y")
#     plt.legend()

def plot_boxes_2d(boxes,name=""):
    plt.figure()
    plt.title(name)
    for i in range(len(boxes)):
        box = boxes[i]
        xL, xU = box[0]
        yL, yU = box[1]
        plt.plot([xL,xL,xU,xU,xL],[yL,yU,yU,yL,yL],'k')
    plt.show()
    
def plot_2d_boxes_traj(boxes, trajs, name=""):
    plt.figure()
    plt.title(name)
    for i in range(len(boxes)):
        box = boxes[i]
        xL, xU = box[0]
        yL, yU = box[1]
        plt.plot([xL,xL,xU,xU,xL],[yL,yU,yU,yL,yL],'k')
    
    for traj in trajs:
        for pt in traj:
            plt.scatter(pt[0], pt[1])
            
    plt.show()
    
# Sampling from polytope

def get_initial(ranges,forwardEuler,constants,remove_ctrl=[]):
    init = []
    initial = random_trajs(ranges,forwardEuler=forwardEuler,constants=constants,n=1)[0]
    init.extend(initial[0])
    final = []
    final_temp = initial[-1]
    for i in range(len(final_temp)):
        if i not in remove_ctrl:
            final.append(final_temp[i])
    init.extend(final)
    return np.array([init]).T

def sample_polytope(initial, N, A, b, proj_dim):
#     points = [initial]
#     for _ in range(N-1):
#         x0 = points[-1]
#         alpha = np.random.normal(0,1,initial.shape)
        
#         theta_min = cvxpy.Variable()
#         objective_min = cvxpy.Minimize(theta_min)
#         constraints = [A*(x0 + theta_min*alpha) <= b]
#         problem_min = cvxpy.Problem(objective_min,constraints)
#         problem_min.solve(solver=cvxpy.GUROBI)
#         theta_min = theta_min.value
        
#         theta_max = cvxpy.Variable()
#         objective_max = cvxpy.Maximize(theta_max)
#         constraints = [A*(x0 + theta_max*alpha) <= b]
#         problem_max = cvxpy.Problem(objective_max,constraints)
#         problem_max.solve(solver=cvxpy.GUROBI)
#         theta_max = theta_max.value
        
#         theta = np.random.uniform(theta_min, theta_max)
#         x_next = x0 + theta*alpha
#         points.append(x_next)

    points = [initial]
    for _ in range(N-1):
        points.append(polytope_sample(A,b,initial))
        
    projected_points = []
    for point in points:
        projected_points.append(point[proj_dim])
        
    return projected_points, points

def verify_polytope(points, A, b):
    valid = []
    for point in points:
        valid.append(np.all(A@point <= b))
    return all(valid)

def polytope_sample(A,b,i_point):
    alpha_list = []
    t = np.random.multivariate_normal(np.zeros(A.shape[1]),np.eye(A.shape[1]),size=(1,)).T
    for r in range(len(A)):
        numerator = (b[r,0] - np.matmul(A[r,:],i_point)[0])
        denominator = (np.matmul(A[r,:],t)[0])
        alpha = numerator/denominator
        p = alpha*t + i_point
        if((np.matmul(A,p) - b <= 1e-9).all()):
            alpha_list.append(alpha)
        if(len(alpha_list) == 2):
            break;
    if(len(alpha_list) < 2):
        print("Error! Less than two intersection points")
    alpha_sorted = np.sort(alpha_list)
    alpha_final = np.random.uniform(alpha_sorted[0],alpha_sorted[1])
    return alpha_final*t + i_point

# Outer Approximation

def outer_approximation(n,A,b,offset=0,start_idx=0):
    dirs = []
    for t in np.linspace(0,np.pi,n):
#         l = [np.cos(t),np.sin(t)]
        l = [0 for _ in range(A.shape[1])]
        l[start_idx] = np.cos(t)
        l[start_idx+1] = np.sin(t)
        dirs.append(np.array([l]).T)
    dirs_ret = []
    alphas = []
    points = []
    for d in dirs:
        x_min = cvxpy.Variable((A.shape[1],1))
        objective_min = cvxpy.Minimize(x_min.T@d)
        constraints_min = [A@x_min <= b]
        problem_min = cvxpy.Problem(objective_min,constraints_min)
        alpha_min = problem_min.solve()
        
        dirs_ret.append(d)
        alphas.append(alpha_min)
        x_min_val = x_min.value
        points.append([x_min_val[start_idx,0],x_min_val[start_idx+1,0]])
        
        x_max = cvxpy.Variable((A.shape[1],1))
        objective_max = cvxpy.Maximize(x_max.T@d)
        constraints_max = [A@x_max <= b]
        problem_max = cvxpy.Problem(objective_max,constraints_max)
        alpha_max = problem_max.solve()
        
        dirs_ret.append(d)
        alphas.append(alpha_max)
        x_max_val = x_max.value
        points.append([x_max_val[start_idx,0],x_max_val[start_idx+1,0]])
    
    return dirs_ret, alphas, points

def outer_approximation_cvx(n,variables,constraints,offset=0):
    dirs = []
    for t in np.linspace(0,np.pi,n):
        l = [np.cos(t),np.sin(t)]
        dirs.append(np.array([l]).T)
    dirs_ret = []
    alphas = []
    points = []
    x_T = variables[-1]["x"]
    y_T = variables[-1]["y"]
    
    for d in dirs:
        objective_min = cvxpy.Minimize(d[0,0]*x_T + d[1,0]*y_T)
        constraints_min = constraints
        problem_min = cvxpy.Problem(objective_min,constraints_min)
        alpha_min = problem_min.solve()
        
        dirs_ret.append(d)
        alphas.append(alpha_min)
        x_min_val = x_T.value
        y_min_val = y_T.value
        points.append([x_min_val,y_min_val])
        
        objective_max = cvxpy.Maximize(d[0,0]*x_T + d[1,0]*y_T)
        constraints_max = constraints
        problem_max = cvxpy.Problem(objective_max,constraints_max)
        alpha_max = problem_max.solve()
        
        dirs_ret.append(d)
        alphas.append(alpha_max)
        x_max_val = x_T.value
        y_max_val = y_T.value
        points.append([x_max_val,y_max_val])
    
    return dirs_ret, alphas, points

def plot_hyperplanes(dirs,alphas,points,num_points,start_idx=0):
    plt.figure()
    min_x, max_x = min([pt[0] for pt in points]), max([pt[0] for pt in points])
    min_y, max_y = min([pt[1] for pt in points]), max([pt[1] for pt in points])
    dev_x, dev_y = 0.1*(max_x-min_x), 0.1*(max_y-min_y)
    x_range = np.linspace(min_x - dev_x,max_x + dev_x,num_points)
    y_range = np.linspace(min_y - dev_y,max_y + dev_y,num_points)
    for i in range(len(dirs)):
        d = dirs[i]
        alpha = alphas[i]
        if d[start_idx+1] == 0:
            c = alpha/d[start_idx]
            ys = list(y_range)
            xs = [c for _ in range(len(ys))]
        else:
            xs = list(x_range)
            ln = lambda x: (-1*d[start_idx]*x + alpha)/d[start_idx+1]
            ys = [ln(x) for x in xs]
        plt.plot(xs,ys)
    plt.xlim([min(x_range),max(x_range)])
    plt.ylim([min(y_range),max(y_range)])
    
def scatter_plot(points):
    plt.figure()
    for pt in points:
        plt.scatter(pt[0], pt[1])

def plot_outer_approximation(n,A,b,num_points,start_idx=0):
    dirs, alphas, points = outer_approximation(n,A,b,start_idx=start_idx)
    plot_hyperplanes(dirs, alphas, points, num_points, start_idx=start_idx)
    scatter_plot(points)

def plot_outer_approximation_cvx(n,variables,constraints,num_points):
    dirs, alphas, points = outer_approximation_cvx(n,variables,constraints)
    plot_hyperplanes(dirs, alphas, points, num_points)
    scatter_plot(points)