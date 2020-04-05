import turtle
import math
import numpy as np
from turtle import *
import time
from queue import LifoQueue

# method to create pattern for turtle
def lsystem(axioms, rules, iterations):
    #  Loop by argument iterations
    for _ in range(iterations):
        # variable that represents newly added axioms pre iteration
        newAxioms = ''

        #   Iterate over axioms parameter, if there is rule for given axiom, it will be added, otherwise whole axiom is added
        for axiom in axioms:
            if axiom in rules:
                newAxioms += rules[axiom]
            else:
                newAxioms += axiom

        # changing value of axioms by newly created axioms
        axioms = newAxioms
    return axioms

# method to move with turtle
def move(axioms_result, angle, default_length):
    # j variable is needed to index Stacks, that are for positions and headings(angles), I used 'stack' called LifoQueue
    # these stacks are used to save actual position and angle if there is any parenthesis in pattern
    # that signals block that after its end, turtle will have to get back to location before parenthesis
    # after that there are turtle initialization commands

    j = 0
    positions = LifoQueue()
    headings = LifoQueue()
    t = turtle.Turtle()  # create the turtle
    wn = turtle.Screen()
    t.right(0)
    t.speed(15)
    # loop to iterate over created result from 'lsystem' method, it checks for its characters and if
    # it equals to some char in any condition it behaves by body of condition
    #F => forward, + => increase angle, - => decrease angle, [ => save actual position & angle to stack, ] => get most recent position & angle from stack and delete it
    for i in range(len(str(axioms_result))):
        if axioms_result[i] == "F":
            t.forward(default_length)
        elif axioms_result[i] == "+":
            t.right(angle)
        elif axioms_result[i] == "-":
            t.left(angle)
        elif axioms_result[i] == "[":
            positions.put(t.position())
            headings.put(t.heading())
            j += 1
        elif axioms_result[i] == "]":
            j -= 1
            t.setposition(positions.get())
            t.setheading(headings.get())

    time.sleep(5)


# First axiom
axiom1 = {"F+F+F+F"}
rules1 = {"F": "F+F-F-FF+F+F-F"}
angle = 90
number_of_iterations_by_axiom = 4
result = lsystem(axiom1, rules1, number_of_iterations_by_axiom)
move(result, angle, 4)

# # Second axiom
# axiom2 = {"F++F++F"}
# rules2 = {"F": "F+F--F+F"}
# angle = 60
# number_of_iterations_by_axiom = 3
# result = lsystem(axiom2, rules2, number_of_iterations_by_axiom)
# move(result, angle, 9)

# # Third axiom
# axiom3 = {"F"}
# rules3 = {"F": "F[+F]F[-F]F"}
# angle = math.degrees(math.pi/7)
# number_of_iterations_by_axiom = 3
# result = lsystem(axiom3, rules3, number_of_iterations_by_axiom)
# move(result, angle, 9)

# # Fourth axiom
# axiom4 = {"F"}
# rules4 = {"F": "FF+[+F-F-F]-[-F+F+F]"}
# angle = math.degrees(math.pi/8)
# number_of_iterations_by_axiom = 2
# result = lsystem(axiom4, rules4, number_of_iterations_by_axiom)
# move(result, angle, 9)
