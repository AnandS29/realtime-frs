#  Copyright (c) 2018, The Regents of the University of California (Regents).
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo   ( vrubies@eecs.berkeley.edu )

import numpy as np
import cvxpy
import matplotlib.pyplot as plt
#from DrawingUtils import plot_extreme_points,graph_plane,PlotPolytope
import random as rnd
import time

# Van Der Pol Oscillator
# 
# x_{t+1} = -y_t \Delta_t + x_t           
# y_{t+1} = -[u_t(1-x_t^2)y_t - x_t] \Delta_t + y_t

T = 30
del_t = -0.1

u_min = -1.0
u_max = 1.0

x_up_b = 0.01 + 1
x_lw_b = -0.01 + 1
y_up_b = 0.01
y_lw_b = -0.01


def McCormick(w,x,y,x_bounds,y_bounds):
    xL,xU = x_bounds
    yL,yU = y_bounds

    c = []
    c.append(w >= xL*y + x*yL - xL*yL)
    c.append(w >= xU*y + x*yU - xU*yU)
    c.append(w <= xU*y + x*yL - xU*yL)
    c.append(w <= x*yU + xL*y - xL*yU)
    return c

Npast = 0
def bounds(x,constraints,Npast=0):

    objective_max = cvxpy.Maximize(x)
    problem_maximum = cvxpy.Problem(objective_max,constraints[-Npast:])
    value_max = problem_maximum.solve(solver=cvxpy.GUROBI)

    objective_min = cvxpy.Minimize(x)
    problem_minimum = cvxpy.Problem(objective_min,constraints[-Npast:])
    value_min = problem_minimum.solve(solver=cvxpy.GUROBI)

    return (value_min,value_max)

def VanDerPolConstraints(T=1,variables=[],constraints=[],control_bounds=[]):

    times = []
    for t in range(T):
        tt = time.time()
        
        x = variables[t]["x"]
        x_bounds = bounds(x,constraints,Npast=Npast)
        y = variables[t]["y"]
        y_bounds = bounds(y,constraints,Npast=Npast)

        ######### lil plot
        xL,xU = x_bounds
        yL,yU = y_bounds
        plt.figure(1)
        if t>0: 
            #plt.clf()
            plt.plot([x_lw_b,x_lw_b,x_up_b,x_up_b,x_lw_b],[y_lw_b,y_up_b,y_up_b,y_lw_b,y_lw_b],'b')
            plt.plot([xL,xL,xU,xU,xL],[yL,yU,yU,yL,yL],'k')
            # print("x_b: ",x_bounds)
            # print("y_b: ",y_bounds)
        plt.pause(1.01)
        ########

        u = variables[t]["u"]
        u_bounds = control_bounds[t]
        constraints.append(u <= u_bounds[1])
        constraints.append(u >= u_bounds[0])

        n = variables[t]["n"]
        constraints.extend(McCormick(n,x,x,x_bounds,x_bounds))

        v = variables[t]["v"]
        constraints.append(v == 1 - n)
        v_bounds = bounds(v,constraints,Npast=Npast)

        w = variables[t]["w"]
        constraints.extend(McCormick(w,y,v,y_bounds,v_bounds))
        w_bounds = bounds(w,constraints,Npast=Npast)        

        z = variables[t]["z"]
        constraints.extend(McCormick(z,u,w,u_bounds,w_bounds)) 

        y_new = variables[t+1]["y"]
        constraints.append(y_new == z*del_t - x*del_t + y)

        x_new = variables[t+1]["x"]
        constraints.append(x_new == y*del_t + x)

        tt = time.time() - tt
        times.append(tt)
        print("Time for ",t,"-th loop: ",tt)

    # plt.figure(2)
    # plt.plot(times,'r*')
    return variables,constraints

variables = [{"u":cvxpy.Variable(),"x":cvxpy.Variable(),"y":cvxpy.Variable(),"z":cvxpy.Variable(),"w":cvxpy.Variable(),"v":cvxpy.Variable(),"n":cvxpy.Variable()} for t in range(T+1)]

plt.figure(1)
plt.plot([x_lw_b,x_lw_b,x_up_b,x_up_b,x_lw_b],[y_lw_b,y_up_b,y_up_b,y_lw_b,y_lw_b],'b')

x = variables[0]["x"]
y = variables[0]["y"]

constraints = []
constraints.append(x >= x_lw_b)
constraints.append(x <= x_up_b)
constraints.append(y >= y_lw_b)
constraints.append(y <= y_up_b)

control_bounds = [(0.2,1.0) for t in range(T)]
#control_bounds = [(2.0,2.5),(2.0,2.5),(2.0,2.5),(2.0,2.5)]

variables,constraints = VanDerPolConstraints(T,variables,constraints,control_bounds)
xL,xU = bounds(variables[-1]["x"],constraints)
yL,yU = bounds(variables[-1]["y"],constraints)

plt.plot([xL,xL,xU,xU,xL],[yL,yU,yU,yL,yL],'r')
plt.pause(1000.0)

