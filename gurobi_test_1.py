# -*- coding: utf-8 -*-

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB

try:

    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')










# This example formulates and solves the following simple QP model:
#  minimize
#      x^2 + x*y + y^2 + y*z + z^2 + 2 x
#  subject to
#      x + 2 y + 3 z >= 4
#      x +   y       >= 1
#      x, y, z non-negative
#
# It solves it once as a continuous model, and once as an integer model.

import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("qp")

# Create variables
x = m.addVar(ub=1.0, name="x")
# y = m.addVar(ub=1.0, name="y")
# z = m.addVar(ub=1.0, name="z")

y = m.addVar(ub=0.5, name="y")
z = m.addVar(lb=1.0, name="z")


# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
#obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
obj = x*x + x*y + y*y + y*z + z*z + 2*x
m.setObjective(obj)

# Add constraint: x + 2 y + 3 z <= 4
m.addConstr(x + 2 * y + 3 * z >= 4, "c0")

# Add constraint: x + y >= 1
m.addConstr(x + y >= 1, "c1")

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())

# As integer model

x.vType = GRB.INTEGER
y.vType = GRB.INTEGER
z.vType = GRB.INTEGER

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())







# This example formulates and solves the following simple QCP model:
#  maximize    x
#  subject to  x + y + z = 1
#              x^2 + y^2 <= z^2 (second-order cone)
#              x^2 <= yz        (rotated second-order cone)
#              x, y, z non-negative

import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("qcp")

# Create variables
x = m.addVar(name="x")
y = m.addVar(name="y")
z = m.addVar(name="z")

# Set objective: x
obj = 1.0*x
m.setObjective(obj, GRB.MAXIMIZE)

# Add constraint: x + y + z = 1
m.addConstr(x + y + z == 1, "c0")

# Add second-order cone: x^2 + y^2 <= z^2
#m.addConstr(x**2 + y**2 <= z**2, "qc0")
m.addConstr(x*x + y*y <= z*z, "qc0")

# Add rotated cone: x^2 <= yz
#m.addConstr(x**2 <= y*z, "qc1")
m.addConstr(x*x <= y*z, "qc1")

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())











# This example creates a very simple Special Ordered Set (SOS) model.
# The model consists of 3 continuous variables, no linear constraints,
# and a pair of SOS constraints of type 1.


import gurobipy as gp
from gurobipy import GRB

try:

    # Create a new model

    model = gp.Model("sos")

    # Create variables

    x0 = model.addVar(ub=1.0, name="x0")
    x1 = model.addVar(ub=1.0, name="x1")
    x2 = model.addVar(ub=2.0, name="x2")

    # Set objective
    model.setObjective(2 * x0 + x1 + x2, GRB.MAXIMIZE)

    # Add first SOS: x0 = 0 or x1 = 0
    model.addSOS(GRB.SOS_TYPE1, [x0, x1], [1, 2])

    # Add second SOS: x0 = 0 or x2 = 0
    model.addSOS(GRB.SOS_TYPE1, [x0, x2], [1, 2])

    model.optimize()

    for v in model.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % model.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')










# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple QP model:
#
#    minimize    x + y + x^2 + x*y + y^2 + y*z + z^2
#    subject to  x + 2 y + 3 z >= 4
#                x +   y       >= 1
#                x, y, z non-negative
#
# The example illustrates the use of dense matrices to store A and Q
# (and dense vectors for the other relevant data).  We don't recommend
# that you use dense matrices, but this example may be helpful if you
# already have your data in this format.

import sys
import gurobipy as gp
from gurobipy import GRB


def dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype,
                   solution):

    model = gp.Model()

    # Add variables to model
    vars = []
    for j in range(cols):
        vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

    # Populate A matrix
    for i in range(rows):
        expr = gp.LinExpr()
        for j in range(cols):
            if A[i][j] != 0:
                expr += A[i][j]*vars[j]
        model.addLConstr(expr, sense[i], rhs[i])

    # Populate objective
    obj = gp.QuadExpr()
    for i in range(cols):
        for j in range(cols):
            if Q[i][j] != 0:
                obj += Q[i][j]*vars[i]*vars[j]
    for j in range(cols):
        if c[j] != 0:
            obj += c[j]*vars[j]
    model.setObjective(obj)

    # Solve
    model.optimize()

    # Write model to a file
    model.write('dense.lp')

    if model.status == GRB.OPTIMAL:
        x = model.getAttr('x', vars)
        for i in range(cols):
            solution[i] = x[i]
        return True
    else:
        return False


# Put model data into dense matrices

c = [1, 1, 0]
Q = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
A = [[1, 2, 3], [1, 1, 0]]
sense = [GRB.GREATER_EQUAL, GRB.GREATER_EQUAL]
rhs = [4, 1]
lb = [0, 0, 0]
ub = [GRB.INFINITY, GRB.INFINITY, GRB.INFINITY]
vtype = [GRB.CONTINUOUS, GRB.CONTINUOUS, GRB.CONTINUOUS]
sol = [0]*3

# Optimize

success = dense_optimize(2, 3, c, Q, A, sense, rhs, lb, ub, vtype, sol)

if success:
    print('x: %g, y: %g, z: %g' % (sol[0], sol[1], sol[2]))
    
    
    
    

# In this example we show the use of general constraints for modeling
# some common expressions. We use as an example a SAT-problem where we
# want to see if it is possible to satisfy at least four (or all) clauses
# of the logical for
#
# L = (x0 or ~x1 or x2)  and (x1 or ~x2 or x3)  and
#     (x2 or ~x3 or x0)  and (x3 or ~x0 or x1)  and
#     (~x0 or ~x1 or x2) and (~x1 or ~x2 or x3) and
#     (~x2 or ~x3 or x0) and (~x3 or ~x0 or x1)
#
# We do this by introducing two variables for each literal (itself and its
# negated value), a variable for each clause, and then two
# variables for indicating if we can satisfy four, and another to identify
# the minimum of the clauses (so if it is one, we can satisfy all clauses)
# and put these two variables in the objective.
# i.e. the Objective function will be
#
# maximize Obj0 + Obj1
#
#  Obj0 = MIN(Clause1, ... , Clause8)
#  Obj1 = 1 -> Clause1 + ... + Clause8 >= 4
#
# thus, the objective value will be two if and only if we can satisfy all
# clauses; one if and only if at least four clauses can be satisfied, and
# zero otherwise.

import gurobipy as gp
from gurobipy import GRB
import sys

try:
    NLITERALS = 4

    n = NLITERALS

    # Example data:
    #   e.g. {0, n+1, 2} means clause (x0 or ~x1 or x2)
    Clauses = [[  0, n+1, 2],
               [  1, n+2, 3],
               [  2, n+3, 0],
               [  3, n+0, 1],
               [n+0, n+1, 2],
               [n+1, n+2, 3],
               [n+2, n+3, 0],
               [n+3, n+0, 1]]

    # Create a new model
    model = gp.Model("Genconstr")

    # initialize decision variables and objective
    Lit = model.addVars(NLITERALS, vtype=GRB.BINARY, name="X")
    NotLit = model.addVars(NLITERALS, vtype=GRB.BINARY, name="NotX")

    Cla = model.addVars(len(Clauses), vtype=GRB.BINARY, name="Clause")

    Obj0 = model.addVar(vtype=GRB.BINARY, name="Obj0")
    Obj1 = model.addVar(vtype=GRB.BINARY, name="Obj1")

    # Link Xi and notXi
    model.addConstrs((Lit[i] + NotLit[i] == 1.0 for i in range(NLITERALS)),
                     name="CNSTR_X")

    # Link clauses and literals
    for i, c in enumerate(Clauses):
        clause = []
        for l in c:
            if l >= n:
                clause.append(NotLit[l-n])
            else:
                clause.append(Lit[l])
        model.addConstr(Cla[i] == gp.or_(clause), "CNSTR_Clause" + str(i))

    # Link objs with clauses
    model.addConstr(Obj0 == gp.min_(Cla), name="CNSTR_Obj0")
    model.addConstr((Obj1 == 1) >> (Cla.sum() >= 4.0), name="CNSTR_Obj1")

    # Set optimization objective
    model.setObjective(Obj0 + Obj1, GRB.MAXIMIZE)

    # Save problem
    model.write("genconstr.mps")
    model.write("genconstr.lp")

    # Optimize
    model.optimize()

    # Status checking
    status = model.getAttr(GRB.Attr.Status)

    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print("The model cannot be solved because it is infeasible or "
              "unbounded")
        sys.exit(1)

    if status != GRB.OPTIMAL:
        print("Optimization was stopped with status ", status)
        sys.exit(1)

    # Print result
    objval = model.getAttr(GRB.Attr.ObjVal)

    if objval > 1.9:
        print("Logical expression is satisfiable")
    elif objval > 0.9:
        print("At least four clauses can be satisfied")
    else:
        print("Not even three clauses can be satisfied")

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

    




# Want to cover three different sets but subject to a common budget of
# elements allowed to be used. However, the sets have different priorities to
# be covered; and we tackle this by using multi-objective optimization.

import gurobipy as gp
from gurobipy import GRB
import sys

try:
    # Sample data
    Groundset = range(20)
    Subsets = range(4)
    Budget = 12
    Set = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]]
    SetObjPriority = [3, 2, 2, 1]
    SetObjWeight = [1.0, 0.25, 1.25, 1.0]

    # Create initial model
    model = gp.Model('multiobj')

    # Initialize decision variables for ground set:
    # x[e] == 1 if element e is chosen for the covering.
    Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')

    # Constraint: limit total number of elements to be picked to be at most
    # Budget
    model.addConstr(Elem.sum() <= Budget, name='Budget')

    # Set global sense for ALL objectives
    model.ModelSense = GRB.MAXIMIZE

    # Limit how many solutions to collect
    model.setParam(GRB.Param.PoolSolutions, 100)

    # Set and configure i-th objective
    for i in Subsets:
        objn = sum(Elem[k]*Set[i][k] for k in range(len(Elem)))
        model.setObjectiveN(objn, i, SetObjPriority[i], SetObjWeight[i],
                            1.0 + i, 0.01, 'Set' + str(i))

    # Save problem
    model.write('multiobj.lp')

    # Optimize
    model.optimize()

    model.setParam(GRB.Param.OutputFlag, 0)

    # Status checking
    status = model.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print("The model cannot be solved because it is infeasible or "
              "unbounded")
        sys.exit(1)

    if status != GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        sys.exit(1)

    # Print best selected set
    print('Selected elements in best solution:')
    selected = [e for e in Groundset if Elem[e].X > 0.9]
    print(" ".join("El{}".format(e) for e in selected))

    # Print number of solutions stored
    nSolutions = model.SolCount
    print('Number of solutions found: ' + str(nSolutions))

    # Print objective values of solutions
    if nSolutions > 10:
        nSolutions = 10
    print('Objective values for first ' + str(nSolutions) + ' solutions:')
    for i in Subsets:
        model.setParam(GRB.Param.ObjNumber, i)
        objvals = []
        for e in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber, e)
            objvals.append(model.ObjNVal)

        print('\tSet{} {:6g} {:6g} {:6g}'.format(i, *objvals))

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError as e:
    print('Encountered an attribute error: ' + str(e))
    
    
    
    
    
    
    
    
    
# This example considers the following separable, convex problem:
#
#   minimize    f(x) - y + g(z)
#   subject to  x + 2 y + 3 z <= 4
#               x +   y       >= 1
#               x,    y,    z <= 1
#
# where f(u) = exp(-u) and g(u) = 2 u^2 - 4 u, for all real u. It
# formulates and solves a simpler LP model by approximating f and
# g with piecewise-linear functions. Then it transforms the model
# into a MIP by negating the approximation for f, which corresponds
# to a non-convex piecewise-linear function, and solves it again.

import gurobipy as gp
from math import exp


def f(u):
    return exp(-u)


def g(u):
    return 2 * u * u - 4 * u


try:

    # Create a new model

    m = gp.Model()

    # Create variables

    lb = 0.0
    ub = 1.0

    x = m.addVar(lb, ub, name='x')
    y = m.addVar(lb, ub, name='y')
    z = m.addVar(lb, ub, name='z')

    # Set objective for y

    m.setObjective(-y)

    # Add piecewise-linear objective functions for x and z

    npts = 101
    ptu = []
    ptf = []
    ptg = []

    for i in range(npts):
        ptu.append(lb + (ub - lb) * i / (npts - 1))
        ptf.append(f(ptu[i]))
        ptg.append(g(ptu[i]))

    m.setPWLObj(x, ptu, ptf)
    m.setPWLObj(z, ptu, ptg)

    # Add constraint: x + 2 y + 3 z <= 4

    m.addConstr(x + 2 * y + 3 * z <= 4, 'c0')

    # Add constraint: x + y >= 1

    m.addConstr(x + y >= 1, 'c1')

    # Optimize model as an LP

    m.optimize()

    print('IsMIP: %d' % m.IsMIP)
    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))
    print('Obj: %g' % m.ObjVal)
    print('')

    # Negate piecewise-linear objective function for x

    for i in range(npts):
        ptf[i] = -ptf[i]

    m.setPWLObj(x, ptu, ptf)

    # Optimize model as a MIP

    m.optimize()

    print('IsMIP: %d' % m.IsMIP)
    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))
    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
    
    
    



# We find alternative epsilon-optimal solutions to a given knapsack
# problem by using PoolSearchMode

from __future__ import print_function
import gurobipy as gp
from gurobipy import GRB
import sys

try:
    # Sample data
    Groundset = range(10)
    objCoef = [32, 32, 15, 15, 6, 6, 1, 1, 1, 1]
    knapsackCoef = [16, 16,  8,  8, 4, 4, 2, 2, 1, 1]
    Budget = 33

    # Create initial model
    model = gp.Model("poolsearch")

    # Create dicts for tupledict.prod() function
    objCoefDict = dict(zip(Groundset, objCoef))
    knapsackCoefDict = dict(zip(Groundset, knapsackCoef))

    # Initialize decision variables for ground set:
    # x[e] == 1 if element e is chosen
    Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')

    # Set objective function
    model.ModelSense = GRB.MAXIMIZE
    model.setObjective(Elem.prod(objCoefDict))

    # Constraint: limit total number of elements to be picked to be at most
    # Budget
    model.addConstr(Elem.prod(knapsackCoefDict) <= Budget, name='Budget')

    # Limit how many solutions to collect
    model.setParam(GRB.Param.PoolSolutions, 1024)
    # Limit the search space by setting a gap for the worst possible solution
    # that will be accepted
    model.setParam(GRB.Param.PoolGap, 0.10)
    # do a systematic search for the k-best solutions
    model.setParam(GRB.Param.PoolSearchMode, 2)

    # save problem
    model.write('poolsearch.lp')

    # Optimize
    model.optimize()

    model.setParam(GRB.Param.OutputFlag, 0)

    # Status checking
    status = model.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print('The model cannot be solved because it is infeasible or '
              'unbounded')
        sys.exit(1)

    if status != GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        sys.exit(1)

    # Print best selected set
    print('Selected elements in best solution:')
    print('\t', end='')
    for e in Groundset:
        if Elem[e].X > .9:
            print(' El%d' % e, end='')
    print('')

    # Print number of solutions stored
    nSolutions = model.SolCount
    print('Number of solutions found: ' + str(nSolutions))

    # Print objective values of solutions
    for e in range(nSolutions):
        model.setParam(GRB.Param.SolutionNumber, e)
        print('%g ' % model.PoolObjVal, end='')
        if e % 15 == 14:
            print('')
    print('')

    # print fourth best set if available
    if (nSolutions >= 4):
        model.setParam(GRB.Param.SolutionNumber, 3)

        print('Selected elements in fourth best solution:')
        print('\t', end='')
        for e in Groundset:
            if Elem[e].Xn > .9:
                print(' El%d' % e, end='')
        print('')

except gp.GurobiError as e:
    print('Gurobi error ' + str(e.errno) + ": " + str(e.message))

except AttributeError as e:
    print('Encountered an attribute error: ' + str(e))
