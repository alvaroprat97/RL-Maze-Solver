import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from environment import Environment
import copy

###################
# Q-FUNCTION GRID #
###################

def make_grid(img):
    line_color = (255,255,255)
    nboxes = 10
    bits_box = int(np.shape(img)[0]/nboxes)
    thickness = 15
    for x in range(1,10):
        point1 = (x*bits_box,0)
        point2 = (x*bits_box,np.shape(img)[0])
        img = cv2.line(img,point1,point2,line_color,thickness)
        img = cv2.line(img,(point1[1],point1[0]),(point2[1],point2[0]),line_color,thickness)
    return img

def create_states(dqn, pixel_box = 50*4,img_size = 500*4):
    states = {'tpos':[],'ppos':[],'predictions':{'N':[],'E':[],'S':[],'W':[]}, 'norm_preds':{'N':[],'E':[],'S':[],'W':[]}}
    for x in range(10):
        x_true_pos = 0.05 + 0.1*x
        x_pixel_pos = int(np.rint(x_true_pos*img_size))
        for y in range(10):
            y_true_pos = 0.05 + 0.1*y
            y_pixel_pos = int(np.rint(500*4 - y_true_pos*img_size))
            states['tpos'].append((x_true_pos,y_true_pos))

            tensor = torch.tensor([x_true_pos,y_true_pos])
            states_tensor = torch.unsqueeze(tensor,0)

            states['ppos'].append((x_pixel_pos,y_pixel_pos))
            preds = dqn.q_network.forward(states_tensor)[0]

            max_p = torch.max(preds).item()
            min_p = torch.min(preds).item()
            for i,direction in enumerate("N E S W".split()):
                states['predictions'][direction].append(preds[i].item())
                states['norm_preds'][direction].append((preds[i].item()-min_p)/(max_p-min_p))
    return states

# TUPLE CENTRE POINT (0.05,0.55)
def get_poly_points(centre_point):
    half_pixel_box = 25*4
    img_size = 500*4
    TL = (centre_point[0] - half_pixel_box, centre_point[1] - half_pixel_box)
    TR = (centre_point[0] + half_pixel_box, centre_point[1] - half_pixel_box)
    BL = (centre_point[0] - half_pixel_box, centre_point[1] + half_pixel_box)
    BR = (centre_point[0] + half_pixel_box, centre_point[1] + half_pixel_box)
    Np = np.array((TL,centre_point,TR))
    Ep = np.array((TR,centre_point,BR))
    Sp = np.array((BR,centre_point,BL))
    Wp = np.array((BL,centre_point,TL))
    return {'N':Np,'E':Ep,'S':Sp,'W':Wp}

def make_color(norm_preds):
    yellow = (255,255,0)
    blue = (0,0,255)
    color_diction = {}
    for i,key in enumerate('N E S W'.split()):
        color_diction[key] = np.add(np.multiply(norm_preds[i],yellow),np.multiply(1-norm_preds[i],blue)).astype(int)
    return color_diction

def run(dqn):
    bit_size = 2000
    img = np.zeros((bit_size,bit_size,3),dtype=np.uint8)
    img = make_grid(img)

    states = create_states(dqn)
    for cell in range(len(states['ppos'])):
        centre_point = states['ppos'][cell]
        poly_dict = get_poly_points(centre_point)

        norm_preds = [states['norm_preds'][direction][cell] for direction in 'N E S W'.split()]
        for i, move in enumerate('N E S W'.split()):
            direction = poly_dict[move]
            c = tuple([int(x) for x in make_color(norm_preds)[move]])
            cv2.fillConvexPoly(img,direction,color = c)
            cv2.polylines(img,[direction],False,color = (0,0,0), thickness = 10)

    img = make_grid(img)
    return img

###############
# GREEDY GRID #
###############

def draw_circle(img, bit_size, agent_state, agent_color):
    agent_centre = (int(agent_state[0] * bit_size), int((1 - agent_state[1]) * bit_size))
    agent_radius = int(0.03 * bit_size)
    cv2.circle(img, agent_centre, agent_radius, agent_color, cv2.FILLED)
    return 0

def find_color(step, max_steps):
    green = (0,255,0)
    red = (255,0,0)
    color = np.add(np.multiply(green,(step/max_steps)),np.multiply(red,(1-step/max_steps))).astype(int)
    c = tuple([int(x) for x in color])
    return c

def run_greedy(agent, dqn, bit_size = 2000, box_size = 200, total_steps = 20, visualise = True):

    gimg = np.zeros((bit_size,bit_size,3),dtype=np.uint8)
    gimg = make_grid(gimg)

    copy_agent = copy.copy(agent)
    copy_agent.reset()

    states = []
    total_rewards = []
    distance_to_final = []
    # IT ACTUALLY TAKES 13 MOVEMENTS TO GET THERE FOR OPTIMAL...!!

    for step in range(total_steps):
        next_state = copy_agent.epsilon_greedy_step(dqn,act_greedy = True)[0]
        states.append(next_state)
        total_rewards.append(copy_agent.total_reward)
        dist_to_goal = np.linalg.norm(next_state - copy_agent.environment.goal_state)
        distance_to_final.append(dist_to_goal)

    if visualise:

        draw_circle(gimg, 2000, states[0], (255,0,0))

        max_steps = len(states)-1
        for step, state in enumerate(states[:-1]):

            c = find_color(step,max_steps)

            x_pos_0 = int(np.rint(states[step][0]*bit_size))
            x_pos_1 = int(np.rint(states[step+1][0]*bit_size))

            y_pos_0 = int(np.rint(bit_size - states[step][1]*bit_size))
            y_pos_1 = int(np.rint(bit_size - states[step+1][1]*bit_size))

            cv2.line(gimg,(x_pos_0,y_pos_0),(x_pos_1,y_pos_1),c,thickness = 18)

        draw_circle(gimg, 2000, states[-1], (0,255,0))

    return gimg, total_rewards, distance_to_final
