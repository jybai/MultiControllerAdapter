# this file is to train and test NN controller
# Invariant of Bernstein polynomial approximation is also shown here whose computation is referred to
# files in ./mat folder, where value-based method and polySOS are used
import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from interval import Interval
from env import Osillator
import scipy.io as io
from scipy.interpolate import interp2d

env = Osillator()

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
	sys.path.append(module_path)
from Agent import Agent

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def save_model(i_episode):
	print("Model Save...")
	if i_episode >= 2400:
		torch.save(agent.actor_local.state_dict(), './actors/actor_'+str(i_episode)+ '.pth')

# train controller for the env system
def ddpg(n_episodes=10000, max_t=200, print_every=1, save_every=200):
	mkdir('./actors')
	scores_deque = deque(maxlen=100)
	scores = []
	
	for i_episode in range(1, n_episodes+1):
		state = env.reset()
		agent.reset()
		score = 0
		timestep = time.time()
		for t in range(max_t):
			action = agent.act(state)[0]
			next_state, reward, done = env.step(action, smoothness=1)
			agent.step(state, action, reward, next_state, done, t)
			score += reward
			state = next_state            
			if done:
				break 
				
		scores_deque.append(score)
		scores.append(score)
		score_average = np.mean(scores_deque)
		
		if i_episode % save_every == 0:
			save_model(i_episode)
		
		if i_episode % print_every == 0:
			print('\rEpisode {}, Average Score: {:.2f}, Current Score:{:.2f}, Max: {:.2f}, Min: {:.2f}, Epsilon: {:.2f}, Momery:{:.1f}'\
				  .format(i_episode, score_average,  scores[-1], np.max(scores), np.min(scores), agent.epsilon, len(agent.memory)), end="\n")     
					
	return scores

# random intial state test for safe, unsafe region or 
#test the controlled trajectory for individual controller and Bernstein polynomial approximation
def test(agent, filename, renew, state_list=[], EP_NUM=500, random_initial_test=True, BP=False):
	agent.actor_local.load_state_dict(torch.load(filename))
	safe = []
	unsafe = []
	fuel_list = []
	trajectory = []
	if not random_initial_test:
		assert EP_NUM == 1
	for ep in range(EP_NUM):
		total_reward = 0
		fuel = 0
		if renew:
			while True:
				state = env.reset()
				if where_inv_polySOS(state):
					break
			# state = env.reset()
			state_list.append(state)
		else: 
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		# print(state)
		trajectory.append(state)
		for t in range(201):
			action = agent.act(state, add_noise=False)[0]
			BP_action = individual_Bernstein_polynomial(state)
			if ep == 0:
				print(abs(action-BP_action))
			fuel += 20 * abs(action)
			if BP:
				next_state, reward, done = env.step(BP_action)
			else:
				next_state, reward, done = env.step(action)
			total_reward += reward
			state = next_state
			trajectory.append(state)
			if done:
				break
		if t >= 95:
			fuel_list.append(fuel)
			safe.append(state_list[ep])
		else:
			unsafe.append(state_list[ep]) 
			print(ep, state_list[ep])
	safe = np.array(safe)
	unsafe = np.array(unsafe)
	if random_initial_test:
		plt.scatter(safe[:, 0], safe[:, 1], c='green')
		if unsafe.shape[0] > 0:
			plt.scatter(unsafe[:, 0], unsafe[:, 1], c='red', marker='*')
		plt.savefig(filename+'.png')
	return state_list, fuel_list, np.array(trajectory)

# please refer to ReachNN code for Bernstein Polynomial approximation
def individual_Bernstein_polynomial(state):
	x0 = state[0]
	x1 = state[1]
	# value-based four partition approach
	# if x0 <= 0 and x1 <= 0:#[5, 5] err 0.0567
	# 	y = 0.000928948617280185*x0**5*x1**5 - 0.00916399869757268*x0**5*x1**4*(0.5*x1 + 1.0) + 0.0359492755565422*x0**5*x1**3*(0.5*x1 + 1.0)**2 - 0.0697812831650494*x0**5*x1**2*(0.5*x1 + 1.0)**3 + 0.0663546609086653*x0**5*x1*(0.5*x1 + 1.0)**4 - 0.0242989823616826*x0**5*(0.5*x1 + 1.0)**5 - 0.00901303508196066*x0**4*x1**5*(0.5*x0 + 1.0) + 0.0888179680344847*x0**4*x1**4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0) - 0.345683835873506*x0**4*x1**3*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**2 + 0.662967804739889*x0**4*x1**2*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**3 - 0.620875261092142*x0**4*x1*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**4 + 0.219175156206897*x0**4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**5 + 0.0342929325034368*x0**3*x1**5*(0.5*x0 + 1.0)**2 - 0.335026834570705*x0**3*x1**4*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0) + 1.29510688414621*x0**3*x1**3*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**2 - 2.45425797734338*x0**3*x1**2*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**3 + 2.23577864969541*x0**3*x1*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**4 - 0.747138020536159*x0**3*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**5 - 0.0649982352275076*x0**2*x1**5*(0.5*x0 + 1.0)**3 + 0.608680173858395*x0**2*x1**4*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0) - 2.3099942479049*x0**2*x1**3*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**2 + 4.2916198431204*x0**2*x1**2*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**3 - 3.78526479374207*x0**2*x1*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**4 + 1.13980752730271*x0**2*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**5 + 0.0606156325401482*x0*x1**5*(0.5*x0 + 1.0)**4 - 0.547379326256262*x0*x1**4*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0) + 1.91962458065328*x0*x1**3*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**2 - 3.40266427514137*x0*x1**2*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**3 + 2.81495386449012*x0*x1*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**4 - 0.623037268729621*x0*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**5 - 0.0219364595755122*x1**5*(0.5*x0 + 1.0)**5 + 0.185778797077615*x1**4*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0) - 0.580437016636*x1**3*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**2 + 0.787827976122108*x1**2*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**3 - 0.402989675793996*x1*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**4 - 0.0577596421143753*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**5
	#    #y = 0.000928948617280185*x0^5*x1^5 - 0.00916399869757268*x0^5*x1^4*(0.5*x1 + 1.0) + 0.0359492755565422*x0^5*x1^3*(0.5*x1 + 1.0)^2 - 0.0697812831650494*x0^5*x1^2*(0.5*x1 + 1.0)^3 + 0.0663546609086653*x0^5*x1*(0.5*x1 + 1.0)^4 - 0.0242989823616826*x0^5*(0.5*x1 + 1.0)^5 - 0.00901303508196066*x0^4*x1^5*(0.5*x0 + 1.0) + 0.0888179680344847*x0^4*x1^4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0) - 0.345683835873506*x0^4*x1^3*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)^2 + 0.662967804739889*x0^4*x1^2*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)^3 - 0.620875261092142*x0^4*x1*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)^4 + 0.219175156206897*x0^4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)^5 + 0.0342929325034368*x0^3*x1^5*(0.5*x0 + 1.0)^2 - 0.335026834570705*x0^3*x1^4*(0.5*x0 + 1.0)^2*(0.5*x1 + 1.0) + 1.29510688414621*x0^3*x1^3*(0.5*x0 + 1.0)^2*(0.5*x1 + 1.0)^2 - 2.45425797734338*x0^3*x1^2*(0.5*x0 + 1.0)^2*(0.5*x1 + 1.0)^3 + 2.23577864969541*x0^3*x1*(0.5*x0 + 1.0)^2*(0.5*x1 + 1.0)^4 - 0.747138020536159*x0^3*(0.5*x0 + 1.0)^2*(0.5*x1 + 1.0)^5 - 0.0649982352275076*x0^2*x1^5*(0.5*x0 + 1.0)^3 + 0.608680173858395*x0^2*x1^4*(0.5*x0 + 1.0)^3*(0.5*x1 + 1.0) - 2.3099942479049*x0^2*x1^3*(0.5*x0 + 1.0)^3*(0.5*x1 + 1.0)^2 + 4.2916198431204*x0^2*x1^2*(0.5*x0 + 1.0)^3*(0.5*x1 + 1.0)^3 - 3.78526479374207*x0^2*x1*(0.5*x0 + 1.0)^3*(0.5*x1 + 1.0)^4 + 1.13980752730271*x0^2*(0.5*x0 + 1.0)^3*(0.5*x1 + 1.0)^5 + 0.0606156325401482*x0*x1^5*(0.5*x0 + 1.0)^4 - 0.547379326256262*x0*x1^4*(0.5*x0 + 1.0)^4*(0.5*x1 + 1.0) + 1.91962458065328*x0*x1^3*(0.5*x0 + 1.0)^4*(0.5*x1 + 1.0)^2 - 3.40266427514137*x0*x1^2*(0.5*x0 + 1.0)^4*(0.5*x1 + 1.0)^3 + 2.81495386449012*x0*x1*(0.5*x0 + 1.0)^4*(0.5*x1 + 1.0)^4 - 0.623037268729621*x0*(0.5*x0 + 1.0)^4*(0.5*x1 + 1.0)^5 - 0.0219364595755122*x1^5*(0.5*x0 + 1.0)^5 + 0.185778797077615*x1^4*(0.5*x0 + 1.0)^5*(0.5*x1 + 1.0) - 0.580437016636*x1^3*(0.5*x0 + 1.0)^5*(0.5*x1 + 1.0)^2 + 0.787827976122108*x1^2*(0.5*x0 + 1.0)^5*(0.5*x1 + 1.0)^3 - 0.402989675793996*x1*(0.5*x0 + 1.0)^5*(0.5*x1 + 1.0)^4 - 0.0577596421143753*(0.5*x0 + 1.0)^5*(0.5*x1 + 1.0)^5
	# elif x0 > 0 and x1 <= 0:#[5, 5] err 0.0706
	# 	y = 0.000150032123969873*x0**5*x1**5 - 0.00308339119439689*x0**5*x1**4*(0.5*x1 + 1.0) + 0.0164295623165587*x0**5*x1**3*(0.5*x1 + 1.0)**2 - 0.0382896529619009*x0**5*x1**2*(0.5*x1 + 1.0)**3 + 0.0430193573968502*x0**5*x1*(0.5*x1 + 1.0)**4 - 0.0198514695158768*x0**5*(0.5*x1 + 1.0)**5 - 0.000253879367538489*x0**4*x1**5*(1 - 0.5*x0) - 0.0145394309352573*x0**4*x1**4*(1 - 0.5*x0)*(0.5*x1 + 1.0) + 0.128504707298381*x0**4*x1**3*(1 - 0.5*x0)*(0.5*x1 + 1.0)**2 - 0.327604701436284*x0**4*x1**2*(1 - 0.5*x0)*(0.5*x1 + 1.0)**3 + 0.381645250111604*x0**4*x1*(1 - 0.5*x0)*(0.5*x1 + 1.0)**4 - 0.180368510809315*x0**4*(1 - 0.5*x0)*(0.5*x1 + 1.0)**5 - 0.00763621744643007*x0**3*x1**5*(1 - 0.5*x0)**2 + 0.0086057362017819*x0**3*x1**4*(1 - 0.5*x0)**2*(0.5*x1 + 1.0) + 0.238691159214774*x0**3*x1**3*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**2 - 1.02972383540377*x0**3*x1**2*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**3 + 1.29748247447734*x0**3*x1*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**4 - 0.658779058746641*x0**3*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**5 - 0.0287775695337441*x0**2*x1**5*(1 - 0.5*x0)**3 + 0.150032624704958*x0**2*x1**4*(1 - 0.5*x0)**3*(0.5*x1 + 1.0) - 0.0564482137952344*x0**2*x1**3*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**2 - 1.08230617762456*x0**2*x1**2*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**3 + 2.06485615092688*x0**2*x1*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**4 - 1.14545087677975*x0**2*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**5 - 0.0443319446475492*x0*x1**5*(1 - 0.5*x0)**4 + 0.332444885751689*x0*x1**4*(1 - 0.5*x0)**4*(0.5*x1 + 1.0) - 0.785933550928012*x0*x1**3*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**2 + 0.344005620138051*x0*x1**2*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**3 + 1.08511143886861*x0*x1*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**4 - 0.748258780456722*x0*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**5 - 0.0219364595755122*x1**5*(1 - 0.5*x0)**5 + 0.185778797077615*x1**4*(1 - 0.5*x0)**5*(0.5*x1 + 1.0) - 0.580437016636*x1**3*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**2 + 0.787827976122108*x1**2*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**3 - 0.402989675793996*x1*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**4 - 0.0577596421143753*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**5
	#    #y = 0.000150032123969873*x0^5*x1^5 - 0.00308339119439689*x0^5*x1^4*(0.5*x1 + 1.0) + 0.0164295623165587*x0^5*x1^3*(0.5*x1 + 1.0)^2 - 0.0382896529619009*x0^5*x1^2*(0.5*x1 + 1.0)^3 + 0.0430193573968502*x0^5*x1*(0.5*x1 + 1.0)^4 - 0.0198514695158768*x0^5*(0.5*x1 + 1.0)^5 - 0.000253879367538489*x0^4*x1^5*(1 - 0.5*x0) - 0.0145394309352573*x0^4*x1^4*(1 - 0.5*x0)*(0.5*x1 + 1.0) + 0.128504707298381*x0^4*x1^3*(1 - 0.5*x0)*(0.5*x1 + 1.0)^2 - 0.327604701436284*x0^4*x1^2*(1 - 0.5*x0)*(0.5*x1 + 1.0)^3 + 0.381645250111604*x0^4*x1*(1 - 0.5*x0)*(0.5*x1 + 1.0)^4 - 0.180368510809315*x0^4*(1 - 0.5*x0)*(0.5*x1 + 1.0)^5 - 0.00763621744643007*x0^3*x1^5*(1 - 0.5*x0)^2 + 0.0086057362017819*x0^3*x1^4*(1 - 0.5*x0)^2*(0.5*x1 + 1.0) + 0.238691159214774*x0^3*x1^3*(1 - 0.5*x0)^2*(0.5*x1 + 1.0)^2 - 1.02972383540377*x0^3*x1^2*(1 - 0.5*x0)^2*(0.5*x1 + 1.0)^3 + 1.29748247447734*x0^3*x1*(1 - 0.5*x0)^2*(0.5*x1 + 1.0)^4 - 0.658779058746641*x0^3*(1 - 0.5*x0)^2*(0.5*x1 + 1.0)^5 - 0.0287775695337441*x0^2*x1^5*(1 - 0.5*x0)^3 + 0.150032624704958*x0^2*x1^4*(1 - 0.5*x0)^3*(0.5*x1 + 1.0) - 0.0564482137952344*x0^2*x1^3*(1 - 0.5*x0)^3*(0.5*x1 + 1.0)^2 - 1.08230617762456*x0^2*x1^2*(1 - 0.5*x0)^3*(0.5*x1 + 1.0)^3 + 2.06485615092688*x0^2*x1*(1 - 0.5*x0)^3*(0.5*x1 + 1.0)^4 - 1.14545087677975*x0^2*(1 - 0.5*x0)^3*(0.5*x1 + 1.0)^5 - 0.0443319446475492*x0*x1^5*(1 - 0.5*x0)^4 + 0.332444885751689*x0*x1^4*(1 - 0.5*x0)^4*(0.5*x1 + 1.0) - 0.785933550928012*x0*x1^3*(1 - 0.5*x0)^4*(0.5*x1 + 1.0)^2 + 0.344005620138051*x0*x1^2*(1 - 0.5*x0)^4*(0.5*x1 + 1.0)^3 + 1.08511143886861*x0*x1*(1 - 0.5*x0)^4*(0.5*x1 + 1.0)^4 - 0.748258780456722*x0*(1 - 0.5*x0)^4*(0.5*x1 + 1.0)^5 - 0.0219364595755122*x1^5*(1 - 0.5*x0)^5 + 0.185778797077615*x1^4*(1 - 0.5*x0)^5*(0.5*x1 + 1.0) - 0.580437016636*x1^3*(1 - 0.5*x0)^5*(0.5*x1 + 1.0)^2 + 0.787827976122108*x1^2*(1 - 0.5*x0)^5*(0.5*x1 + 1.0)^3 - 0.402989675793996*x1*(1 - 0.5*x0)^5*(0.5*x1 + 1.0)^4 - 0.0577596421143753*(1 - 0.5*x0)^5*(0.5*x1 + 1.0)^5
	# elif x0 <= 0 and x1 > 0:#[5, 5] err 0.099
	# 	y = -1.68237185631749e-5*x0**5*x1**5 - 0.00137265691589281*x0**5*x1**4*(1 - 0.5*x1) - 0.0128738605645146*x0**5*x1**3*(1 - 0.5*x1)**2 - 0.0400875482389648*x0**5*x1**2*(1 - 0.5*x1)**3 - 0.0522590867847153*x0**5*x1*(1 - 0.5*x1)**4 - 0.0242989823616826*x0**5*(1 - 0.5*x1)**5 - 0.000857136995611502*x0**4*x1**5*(0.5*x0 + 1.0) + 0.00365050759497323*x0**4*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0) + 0.0636022984063337*x0**4*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0) + 0.287262593040093*x0**4*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0) + 0.433276773681636*x0**4*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0) + 0.219175156206897*x0**4*(1 - 0.5*x1)**5*(0.5*x0 + 1.0) + 0.00793058201122367*x0**3*x1**5*(0.5*x0 + 1.0)**2 + 0.025894239272097*x0**3*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**2 - 0.0898508510375479*x0**3*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**2 - 0.642500488572981*x0**3*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**2 - 1.29596615986417*x0**3*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**2 - 0.747138020536159*x0**3*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**2 - 0.0299095841895402*x0**2*x1**5*(0.5*x0 + 1.0)**3 - 0.172563602727978*x0**2*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**3 - 0.189230153834663*x0**2*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**3 + 0.484882952902779*x0**2*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**3 + 1.59618721155199*x0**2*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**3 + 1.13980752730271*x0**2*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**3 + 0.0429022583350872*x0*x1**5*(0.5*x0 + 1.0)**4 + 0.328809661605968*x0*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**4 + 0.827482377803221*x0*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**4 + 0.712366052873993*x0*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**4 - 0.184462714809*x0*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**4 - 0.623037268729621*x0*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**4 - 0.0198729702610131*x1**5*(0.5*x0 + 1.0)**5 - 0.172419485487368*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**5 - 0.552964576638355*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**5 - 0.828644185698245*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**5 - 0.523394158704696*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**5 - 0.0577596421143753*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**5
	#    #y = -1.68237185631749e-5*x0^5*x1^5 - 0.00137265691589281*x0^5*x1^4*(1 - 0.5*x1) - 0.0128738605645146*x0^5*x1^3*(1 - 0.5*x1)^2 - 0.0400875482389648*x0^5*x1^2*(1 - 0.5*x1)^3 - 0.0522590867847153*x0^5*x1*(1 - 0.5*x1)^4 - 0.0242989823616826*x0^5*(1 - 0.5*x1)^5 - 0.000857136995611502*x0^4*x1^5*(0.5*x0 + 1.0) + 0.00365050759497323*x0^4*x1^4*(1 - 0.5*x1)*(0.5*x0 + 1.0) + 0.0636022984063337*x0^4*x1^3*(1 - 0.5*x1)^2*(0.5*x0 + 1.0) + 0.287262593040093*x0^4*x1^2*(1 - 0.5*x1)^3*(0.5*x0 + 1.0) + 0.433276773681636*x0^4*x1*(1 - 0.5*x1)^4*(0.5*x0 + 1.0) + 0.219175156206897*x0^4*(1 - 0.5*x1)^5*(0.5*x0 + 1.0) + 0.00793058201122367*x0^3*x1^5*(0.5*x0 + 1.0)^2 + 0.025894239272097*x0^3*x1^4*(1 - 0.5*x1)*(0.5*x0 + 1.0)^2 - 0.0898508510375479*x0^3*x1^3*(1 - 0.5*x1)^2*(0.5*x0 + 1.0)^2 - 0.642500488572981*x0^3*x1^2*(1 - 0.5*x1)^3*(0.5*x0 + 1.0)^2 - 1.29596615986417*x0^3*x1*(1 - 0.5*x1)^4*(0.5*x0 + 1.0)^2 - 0.747138020536159*x0^3*(1 - 0.5*x1)^5*(0.5*x0 + 1.0)^2 - 0.0299095841895402*x0^2*x1^5*(0.5*x0 + 1.0)^3 - 0.172563602727978*x0^2*x1^4*(1 - 0.5*x1)*(0.5*x0 + 1.0)^3 - 0.189230153834663*x0^2*x1^3*(1 - 0.5*x1)^2*(0.5*x0 + 1.0)^3 + 0.484882952902779*x0^2*x1^2*(1 - 0.5*x1)^3*(0.5*x0 + 1.0)^3 + 1.59618721155199*x0^2*x1*(1 - 0.5*x1)^4*(0.5*x0 + 1.0)^3 + 1.13980752730271*x0^2*(1 - 0.5*x1)^5*(0.5*x0 + 1.0)^3 + 0.0429022583350872*x0*x1^5*(0.5*x0 + 1.0)^4 + 0.328809661605968*x0*x1^4*(1 - 0.5*x1)*(0.5*x0 + 1.0)^4 + 0.827482377803221*x0*x1^3*(1 - 0.5*x1)^2*(0.5*x0 + 1.0)^4 + 0.712366052873993*x0*x1^2*(1 - 0.5*x1)^3*(0.5*x0 + 1.0)^4 - 0.184462714809*x0*x1*(1 - 0.5*x1)^4*(0.5*x0 + 1.0)^4 - 0.623037268729621*x0*(1 - 0.5*x1)^5*(0.5*x0 + 1.0)^4 - 0.0198729702610131*x1^5*(0.5*x0 + 1.0)^5 - 0.172419485487368*x1^4*(1 - 0.5*x1)*(0.5*x0 + 1.0)^5 - 0.552964576638355*x1^3*(1 - 0.5*x1)^2*(0.5*x0 + 1.0)^5 - 0.828644185698245*x1^2*(1 - 0.5*x1)^3*(0.5*x0 + 1.0)^5 - 0.523394158704696*x1*(1 - 0.5*x1)^4*(0.5*x0 + 1.0)^5 - 0.0577596421143753*(1 - 0.5*x1)^5*(0.5*x0 + 1.0)^5
	# elif x0 > 0 and x1 > 0:#[5, 5] err 0.096
	# 	y = -0.000861963318768725*x0**5*x1**5 - 0.00838880289222062*x0**5*x1**4*(1 - 0.5*x1) - 0.0326167222357075*x0**5*x1**3*(1 - 0.5*x1)**2 - 0.0613296039754442*x0**5*x1**2*(1 - 0.5*x1)**3 - 0.0559160626338112*x0**5*x1*(1 - 0.5*x1)**4 - 0.0198514695158768*x0**5*(1 - 0.5*x1)**5 - 0.00826681573687151*x0**4*x1**5*(1 - 0.5*x0) - 0.079969372754532*x0**4*x1**4*(1 - 0.5*x0)*(1 - 0.5*x1) - 0.306623718288423*x0**4*x1**3*(1 - 0.5*x0)*(1 - 0.5*x1)**2 - 0.584296310949316*x0**4*x1**2*(1 - 0.5*x0)*(1 - 0.5*x1)**3 - 0.528574914984719*x0**4*x1*(1 - 0.5*x0)*(1 - 0.5*x1)**4 - 0.180368510809315*x0**4*(1 - 0.5*x0)*(1 - 0.5*x1)**5 - 0.0313188237034999*x0**3*x1**5*(1 - 0.5*x0)**2 - 0.29905126993831*x0**3*x1**4*(1 - 0.5*x0)**2*(1 - 0.5*x1) - 1.12719547163565*x0**3*x1**3*(1 - 0.5*x0)**2*(1 - 0.5*x1)**2 - 2.10559425537224*x0**3*x1**2*(1 - 0.5*x0)**2*(1 - 0.5*x1)**3 - 1.95080932166931*x0**3*x1*(1 - 0.5*x0)**2*(1 - 0.5*x1)**4 - 0.658779058746641*x0**3*(1 - 0.5*x0)**2*(1 - 0.5*x1)**5 - 0.0592617106736781*x0**2*x1**5*(1 - 0.5*x0)**3 - 0.551113489069787*x0**2*x1**4*(1 - 0.5*x0)**3*(1 - 0.5*x1) - 2.02194466139783*x0**2*x1**3*(1 - 0.5*x0)**3*(1 - 0.5*x1)**2 - 3.64300900409365*x0**2*x1**2*(1 - 0.5*x0)**3*(1 - 0.5*x1)**3 - 3.28587372300366*x0**2*x1*(1 - 0.5*x0)**3*(1 - 0.5*x1)**4 - 1.14545087677975*x0**2*(1 - 0.5*x0)**3*(1 - 0.5*x1)**5 - 0.0547272187272542*x0*x1**5*(1 - 0.5*x0)**4 - 0.497797167055983*x0*x1**4*(1 - 0.5*x0)**4*(1 - 0.5*x1) - 1.76798959760862*x0*x1**3*(1 - 0.5*x0)**4*(1 - 0.5*x1)**2 - 3.04205965160895*x0*x1**2*(1 - 0.5*x0)**4*(1 - 0.5*x1)**3 - 2.49800606849821*x0*x1*(1 - 0.5*x0)**4*(1 - 0.5*x1)**4 - 0.748258780456722*x0*(1 - 0.5*x0)**4*(1 - 0.5*x1)**5 - 0.0198729702610131*x1**5*(1 - 0.5*x0)**5 - 0.172419485487368*x1**4*(1 - 0.5*x0)**5*(1 - 0.5*x1) - 0.552964576638355*x1**3*(1 - 0.5*x0)**5*(1 - 0.5*x1)**2 - 0.828644185698245*x1**2*(1 - 0.5*x0)**5*(1 - 0.5*x1)**3 - 0.523394158704696*x1*(1 - 0.5*x0)**5*(1 - 0.5*x1)**4 - 0.0577596421143753*(1 - 0.5*x0)**5*(1 - 0.5*x1)**5
	# else:
	# 	raise ValueError('undefined partition for Bernstein polynomial approximation')
	
	# three partiton approximation for SOS SDP, inner invariant computation
	if x0 <= -1: #[3, 3] err 0.04
		y = 0.425211076103333*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)**3*(0.5*x1 + 1)**2 + 1.02365428669482*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)**2*(1.0*x0 + 2.0)*(0.5*x1 + 1)**2 + 2.95504052725062*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**2*(0.5*x1 + 1)**2 + 1.15130885442421*(0.5 - 0.25*x1)*(0.5*x0 + 1)**3*(0.5*x1 + 1)**2 + 0.118905423011864*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)**3 + 0.348304683461052*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)**2*(1.0*x0 + 2.0) + 1.34236809633121*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**2 + 0.856536211071972*(1 - 0.5*x1)**3*(0.5*x0 + 1)**3 + 0.660880486810856*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)**3*(0.25*x1 + 0.5) + 1.89507514178296*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)**2*(1.0*x0 + 2.0)*(0.25*x1 + 0.5) + 7.1125340402736*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**2*(0.25*x1 + 0.5) + 4.32513165638667*(1 - 0.5*x1)**2*(0.5*x0 + 1)**3*(0.25*x1 + 0.5) + 0.00215343597608638*(-1.0*x0 - 1.0)**3*(0.5*x1 + 1)**3 - 0.0259029854064857*(-1.0*x0 - 1.0)**2*(1.0*x0 + 2.0)*(0.5*x1 + 1)**3 - 0.247595868986717*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**2*(0.5*x1 + 1)**3 - 0.283330236206298*(0.5*x0 + 1)**3*(0.5*x1 + 1)**3
	   #y = 0.425211076103333*(0.5 - 0.25*(3*y))*(-1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^2 + 1.02365428669482*(0.5 - 0.25*(3*y))*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0)*(0.5*(3*y) + 1)^2 + 2.95504052725062*(0.5 - 0.25*(3*y))*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2*(0.5*(3*y) + 1)^2 + 1.15130885442421*(0.5 - 0.25*(3*y))*(0.5*(3*x) + 1)^3*(0.5*(3*y) + 1)^2 + 0.118905423011864*(1 - 0.5*(3*y))^3*(-1.0*(3*x) - 1.0)^3 + 0.348304683461052*(1 - 0.5*(3*y))^3*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0) + 1.34236809633121*(1 - 0.5*(3*y))^3*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2 + 0.856536211071972*(1 - 0.5*(3*y))^3*(0.5*(3*x) + 1)^3 + 0.660880486810856*(1 - 0.5*(3*y))^2*(-1.0*(3*x) - 1.0)^3*(0.25*(3*y) + 0.5) + 1.89507514178296*(1 - 0.5*(3*y))^2*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0)*(0.25*(3*y) + 0.5) + 7.1125340402736*(1 - 0.5*(3*y))^2*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2*(0.25*(3*y) + 0.5) + 4.32513165638667*(1 - 0.5*(3*y))^2*(0.5*(3*x) + 1)^3*(0.25*(3*y) + 0.5) + 0.00215343597608638*(-1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^3 - 0.0259029854064857*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0)*(0.5*(3*y) + 1)^3 - 0.247595868986717*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2*(0.5*(3*y) + 1)^3 - 0.283330236206298*(0.5*(3*x) + 1)^3*(0.5*(3*y) + 1)^3
	elif -1 < x0 and x0 <= 1:#[3, 3] err 0.09
		y = -0.243584378266395*(0.5 - 0.5*x0)*(0.5 - 0.25*x1)*(x0 + 1)**2*(0.5*x1 + 1)**2 + 0.0556068616975541*(0.5 - 0.5*x0)*(1 - 0.5*x1)**3*(x0 + 1)**2 + 0.0150306530769696*(0.5 - 0.5*x0)*(1 - 0.5*x1)**2*(x0 + 1)**2*(0.25*x1 + 0.5) - 0.0647859069392533*(0.5 - 0.5*x0)*(x0 + 1)**2*(0.5*x1 + 1)**3 + 0.0179892008503783*(0.5 - 0.25*x1)*(1 - x0)**3*(0.5*x1 + 1)**2 - 0.0566021333753245*(0.5 - 0.25*x1)*(1 - x0)**2*(0.5*x0 + 0.5)*(0.5*x1 + 1)**2 - 0.0572932821028275*(0.5 - 0.25*x1)*(x0 + 1)**3*(0.5*x1 + 1)**2 + 0.0133833782979996*(1 - x0)**3*(1 - 0.5*x1)**3 + 0.0675801821310418*(1 - x0)**3*(1 - 0.5*x1)**2*(0.25*x1 + 0.5) - 0.0044270349407234*(1 - x0)**3*(0.5*x1 + 1)**3 + 0.071725194207449*(1 - x0)**2*(1 - 0.5*x1)**3*(0.5*x0 + 0.5) + 0.272500805218411*(1 - x0)**2*(1 - 0.5*x1)**2*(0.5*x0 + 0.5)*(0.25*x1 + 0.5) - 0.0530354952052034*(1 - x0)**2*(0.5*x0 + 0.5)*(0.5*x1 + 1)**3 + 0.00432268189630805*(1 - 0.5*x1)**3*(x0 + 1)**3 - 0.0291911782061452*(1 - 0.5*x1)**2*(x0 + 1)**3*(0.25*x1 + 0.5) - 0.0121985520699155*(x0 + 1)**3*(0.5*x1 + 1)**3	
	   #y = -0.243584378266395*(0.5 - 0.5*(3*x))*(0.5 - 0.25*(3*y))*((3*x) + 1)^2*(0.5*(3*y) + 1)^2 + 0.0556068616975541*(0.5 - 0.5*(3*x))*(1 - 0.5*(3*y))^3*((3*x) + 1)^2 + 0.0150306530769696*(0.5 - 0.5*(3*x))*(1 - 0.5*(3*y))^2*((3*x) + 1)^2*(0.25*(3*y) + 0.5) - 0.0647859069392533*(0.5 - 0.5*(3*x))*((3*x) + 1)^2*(0.5*(3*y) + 1)^3 + 0.0179892008503783*(0.5 - 0.25*(3*y))*(1 - (3*x))^3*(0.5*(3*y) + 1)^2 - 0.0566021333753245*(0.5 - 0.25*(3*y))*(1 - (3*x))^2*(0.5*(3*x) + 0.5)*(0.5*(3*y) + 1)^2 - 0.0572932821028275*(0.5 - 0.25*(3*y))*((3*x) + 1)^3*(0.5*(3*y) + 1)^2 + 0.0133833782979996*(1 - (3*x))^3*(1 - 0.5*(3*y))^3 + 0.0675801821310418*(1 - (3*x))^3*(1 - 0.5*(3*y))^2*(0.25*(3*y) + 0.5) - 0.0044270349407234*(1 - (3*x))^3*(0.5*(3*y) + 1)^3 + 0.071725194207449*(1 - (3*x))^2*(1 - 0.5*(3*y))^3*(0.5*(3*x) + 0.5) + 0.272500805218411*(1 - (3*x))^2*(1 - 0.5*(3*y))^2*(0.5*(3*x) + 0.5)*(0.25*(3*y) + 0.5) - 0.0530354952052034*(1 - (3*x))^2*(0.5*(3*x) + 0.5)*(0.5*(3*y) + 1)^3 + 0.00432268189630805*(1 - 0.5*(3*y))^3*((3*x) + 1)^3 - 0.0291911782061452*(1 - 0.5*(3*y))^2*((3*x) + 1)^3*(0.25*(3*y) + 0.5) - 0.0121985520699155*((3*x) + 1)^3*(0.5*(3*y) + 1)^3
	elif x0 > 1: #[3, 3] err 0.0415
		y = -3.66677005458096*(0.5 - 0.25*x1)*(1 - 0.5*x0)**3*(0.5*x1 + 1)**2 - 6.19193016324512*(0.5 - 0.25*x1)*(1 - 0.5*x0)**2*(1.0*x0 - 1.0)*(0.5*x1 + 1)**2 - 1.64930671669314*(0.5 - 0.25*x1)*(2.0 - 1.0*x0)*(1.0*x0 - 1.0)**2*(0.5*x1 + 1)**2 - 0.572970428326949*(0.5 - 0.25*x1)*(1.0*x0 - 1.0)**3*(0.5*x1 + 1)**2 + 0.276651641363715*(1 - 0.5*x0)**3*(1 - 0.5*x1)**3 - 1.86823540519329*(1 - 0.5*x0)**3*(1 - 0.5*x1)**2*(0.25*x1 + 0.5) - 0.780707332474591*(1 - 0.5*x0)**3*(0.5*x1 + 1)**3 + 0.209656119344319*(1 - 0.5*x0)**2*(1 - 0.5*x1)**3*(1.0*x0 - 1.0) - 3.51295090211393*(1 - 0.5*x0)**2*(1 - 0.5*x1)**2*(1.0*x0 - 1.0)*(0.25*x1 + 0.5) - 1.22561406884481*(1 - 0.5*x0)**2*(1.0*x0 - 1.0)*(0.5*x1 + 1)**3 - 0.00100089274721571*(1 - 0.5*x1)**3*(2.0 - 1.0*x0)*(1.0*x0 - 1.0)**2 - 0.0192041118681437*(1 - 0.5*x1)**3*(1.0*x0 - 1.0)**3 - 1.0284199907425*(1 - 0.5*x1)**2*(2.0 - 1.0*x0)*(1.0*x0 - 1.0)**2*(0.25*x1 + 0.5) - 0.38295897095889*(1 - 0.5*x1)**2*(1.0*x0 - 1.0)**3*(0.25*x1 + 0.5) - 0.319943178077135*(2.0 - 1.0*x0)*(1.0*x0 - 1.0)**2*(0.5*x1 + 1)**3 - 0.110331304802397*(1.0*x0 - 1.0)**3*(0.5*x1 + 1)**3
	   #y = -3.66677005458096*(0.5 - 0.25*(3*y))*(1 - 0.5*(3*x))^3*(0.5*(3*y) + 1)^2 - 6.19193016324512*(0.5 - 0.25*(3*y))*(1 - 0.5*(3*x))^2*(1.0*(3*x) - 1.0)*(0.5*(3*y) + 1)^2 - 1.64930671669314*(0.5 - 0.25*(3*y))*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2*(0.5*(3*y) + 1)^2 - 0.572970428326949*(0.5 - 0.25*(3*y))*(1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^2 + 0.276651641363715*(1 - 0.5*(3*x))^3*(1 - 0.5*(3*y))^3 - 1.86823540519329*(1 - 0.5*(3*x))^3*(1 - 0.5*(3*y))^2*(0.25*(3*y) + 0.5) - 0.780707332474591*(1 - 0.5*(3*x))^3*(0.5*(3*y) + 1)^3 + 0.209656119344319*(1 - 0.5*(3*x))^2*(1 - 0.5*(3*y))^3*(1.0*(3*x) - 1.0) - 3.51295090211393*(1 - 0.5*(3*x))^2*(1 - 0.5*(3*y))^2*(1.0*(3*x) - 1.0)*(0.25*(3*y) + 0.5) - 1.22561406884481*(1 - 0.5*(3*x))^2*(1.0*(3*x) - 1.0)*(0.5*(3*y) + 1)^3 - 0.00100089274721571*(1 - 0.5*(3*y))^3*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2 - 0.0192041118681437*(1 - 0.5*(3*y))^3*(1.0*(3*x) - 1.0)^3 - 1.0284199907425*(1 - 0.5*(3*y))^2*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2*(0.25*(3*y) + 0.5) - 0.38295897095889*(1 - 0.5*(3*y))^2*(1.0*(3*x) - 1.0)^3*(0.25*(3*y) + 0.5) - 0.319943178077135*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2*(0.5*(3*y) + 1)^3 - 0.110331304802397*(1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^3

	else:
		raise ValueError('undefined partition for approximation')
	return y

def where_inv_valuebased(state):
	invariant = io.loadmat('./mat/inv.mat')['V']
	x_loc = state[0]
	y_loc = state[1]
	x1 = np.linspace(-2.4, 2.4, 240)
	y1 = np.linspace(-2.4, 2.4, 240)
	inv1 = interp2d(x1, y1, invariant, kind='linear')(x_loc, y_loc)
	return inv1<1e-8

def where_inv_polySOS(state):
	x = state[0]
	y = state[1]
	#poly 10, disturbance 0.2
	#inv1 = -0.0186297374616+0.000326042536362*(x/3)+7.87795721704e-05*(y/3)-0.0245403645255*(x/3)**2-0.0145569110953*(x/3)**2*(y/3)-0.059774607903*(x/3)**3+0.000778671180061*(x/3)*(y/3)+0.737707316033*(x/3)**3*(y/3)-0.000508694408648*(y/3)**2-0.00962785698989*(x/3)*(y/3)**2-0.00224906344622*(y/3)**3+0.514446215841*(x/3)**2*(y/3)**2+0.238589081278*(x/3)**2*(y/3)**3+0.343583513402*(x/3)**3*(y/3)**2+0.000509332189018*(x/3)*(y/3)**3+3.49071967837*(x/3)**3*(y/3)**3+2.54848867822*(x/3)**4+0.00742730412341*(y/3)**4+1.04870165096*(x/3)**5+0.141134378896*(x/3)**4*(y/3)+0.14439498584*(x/3)*(y/3)**4+0.0285877824801*(y/3)**5-5.83768040184*(x/3)**6-2.51641802335*(x/3)**5*(y/3)+3.01569530345*(x/3)**4*(y/3)**2+1.73654696876*(x/3)**2*(y/3)**4+1.21217614154*(x/3)*(y/3)**5-0.0639716949012*(y/3)**6-2.25465038144*(x/3)**7-0.40332413766*(x/3)**6*(y/3)-1.44972371159*(x/3)**5*(y/3)**2-0.220769222908*(x/3)**4*(y/3)**3+0.64105658932*(x/3)**3*(y/3)**4-0.577750929415*(x/3)**2*(y/3)**5-0.726679077695*(x/3)*(y/3)**6-0.0900749354262*(y/3)**7+5.74618257194*(x/3)**8+3.92108620987*(x/3)**7*(y/3)-16.3355156347*(x/3)**6*(y/3)**2-11.9611447227*(x/3)**5*(y/3)**3-2.38649911655*(x/3)**4*(y/3)**4-4.36505302442*(x/3)**3*(y/3)**5-4.88659944522*(x/3)**2*(y/3)**6-2.50705929521*(x/3)*(y/3)**7+1.50145382304*(y/3)**8+1.29930633227*(x/3)**9+0.320820263513*(x/3)**8*(y/3)+0.952381536266*(x/3)**7*(y/3)**2+0.0361037148917*(x/3)**6*(y/3)**3+0.869677706192*(x/3)**5*(y/3)**4+0.10685560055*(x/3)**4*(y/3)**5-1.84875451629*(x/3)**3*(y/3)**6+0.397316171519*(x/3)**2*(y/3)**7+0.863141576896*(x/3)*(y/3)**8+0.0771594074445*(y/3)**9-1.93438405021*(x/3)**10-2.38824191419*(x/3)**9*(y/3)+14.6017559257*(x/3)**8*(y/3)**2+8.53939158801*(x/3)**7*(y/3)**3+3.37487447424*(x/3)**6*(y/3)**4+5.40811901492*(x/3)**5*(y/3)**5+4.55612395012*(x/3)**4*(y/3)**6+4.38009060668*(x/3)**3*(y/3)**7-1.01656714635*(x/3)**2*(y/3)**8+0.0129416741164*(x/3)*(y/3)**9
	#poly 8, disturbance 0.1
	#inv1 = -0.185158249313-0.000473119064748*(x/3)+7.140793824e-05*(y/3)+0.00836356573883*(x/3)**2+0.0253138051056*(x/3)**2*(y/3)+0.309155891661*(x/3)**3+0.00465698950305*(x/3)*(y/3)+0.571994549891*(x/3)**3*(y/3)+0.00413435632066*(y/3)**2+0.0394295044232*(x/3)*(y/3)**2-0.000372159449832*(y/3)**3+1.30769403886*(x/3)**2*(y/3)**2+0.111360475954*(x/3)**2*(y/3)**3+0.112562431688*(x/3)**3*(y/3)**2+0.344773262005*(x/3)*(y/3)**3-0.296721683604*(x/3)**3*(y/3)**3+5.49744084087*(x/3)**4-0.15799424306*(y/3)**4-0.766154330592*(x/3)**5-0.119272033703*(x/3)**4*(y/3)-0.25461634796*(x/3)*(y/3)**4-0.0141542652043*(y/3)**5-12.0471160801*(x/3)**6-1.1203197395*(x/3)**5*(y/3)-2.09261930534*(x/3)**4*(y/3)**2-6.88983225391*(x/3)**2*(y/3)**4-1.31921472181*(x/3)*(y/3)**5+4.00683818715*(y/3)**6+0.46954453433*(x/3)**7+0.131881504241*(x/3)**6*(y/3)-0.114360742575*(x/3)**5*(y/3)**2-0.183219552511*(x/3)**4*(y/3)**3-0.0801923298815*(x/3)**3*(y/3)**4+0.0158830858269*(x/3)**2*(y/3)**5+0.209515768697*(x/3)*(y/3)**6+0.00262033253548*(y/3)**7+7.74893172914*(x/3)**8+0.419448892336*(x/3)**7*(y/3)+1.27519142123*(x/3)**6*(y/3)**2-0.113501170791*(x/3)**5*(y/3)**3+4.63195071404*(x/3)**4*(y/3)**4+1.24318178509*(x/3)**3*(y/3)**5+3.46588491504*(x/3)**2*(y/3)**6+0.842679004196*(x/3)*(y/3)**7-3.20186643367*(y/3)**8
	#poly 8, disturbance 0.05
	#inv1 = -0.202365181636-0.000215009023518*(x/3)-1.70396414056e-05*(y/3)+0.0579883929434*(x/3)**2+0.0218489517004*(x/3)**2*(y/3)+0.312256054801*(x/3)**3+0.00667207216021*(x/3)*(y/3)+0.379472029987*(x/3)**3*(y/3)+0.00545236408164*(y/3)**2+0.0473604175964*(x/3)*(y/3)**2+0.00775023891412*(y/3)**3+1.38079717409*(x/3)**2*(y/3)**2+0.0913008695767*(x/3)**2*(y/3)**3-0.0191234259779*(x/3)**3*(y/3)**2+0.847668246321*(x/3)*(y/3)**3-0.573961210864*(x/3)**3*(y/3)**3+3.39494195498*(x/3)**4-0.130120070322*(y/3)**4-0.862579484751*(x/3)**5-0.0621650388833*(x/3)**4*(y/3)-0.311001976429*(x/3)*(y/3)**4-0.0492159781768*(y/3)**5-6.31541418556*(x/3)**6-0.709902456876*(x/3)**5*(y/3)-2.55473582039*(x/3)**4*(y/3)**2-6.71979264816*(x/3)**2*(y/3)**4-3.05080936341*(x/3)*(y/3)**5+4.30896302621*(y/3)**6+0.597053864936*(x/3)**7+0.0232154822223*(x/3)**6*(y/3)+0.0168917541206*(x/3)**5*(y/3)**2-0.032525497937*(x/3)**4*(y/3)**3+0.101338803745*(x/3)**3*(y/3)**4-0.0729773972423*(x/3)**2*(y/3)**5+0.243348816782*(x/3)*(y/3)**6+0.0438214864079*(y/3)**7+3.77676086304*(x/3)**8+0.157443976319*(x/3)**7*(y/3)+1.67738304589*(x/3)**6*(y/3)**2-0.0562328419816*(x/3)**5*(y/3)**3+4.58131723787*(x/3)**4*(y/3)**4+1.74615514317*(x/3)**3*(y/3)**5+3.36050391139*(x/3)**2*(y/3)**6+2.1615856001*(x/3)*(y/3)**7-3.55293798867*(y/3)**8
	#poly 10, disturbance 0.05
	inv1 = -0.263913574764-0.000528242941314*(x/3)+0.000223494053431*(y/3)+0.136880834381*(x/3)**2+0.0453223772422*(x/3)**2*(y/3)+0.451060956722*(x/3)**3+0.0286858606342*(x/3)*(y/3)+0.149573897799*(x/3)**3*(y/3)+0.00286877418857*(y/3)**2+0.0756223492812*(x/3)*(y/3)**2-0.0604702390379*(y/3)**3+2.14093826624*(x/3)**2*(y/3)**2-0.201502301296*(x/3)**2*(y/3)**3-0.275534348368*(x/3)**3*(y/3)**2+1.86352665373*(x/3)*(y/3)**3-8.18848993039*(x/3)**3*(y/3)**3+7.31224043295*(x/3)**4+1.41684160955*(y/3)**4-1.83528241165*(x/3)**5-0.257056130739*(x/3)**4*(y/3)-0.543601992228*(x/3)*(y/3)**4+0.392191892045*(y/3)**5-26.1600398701*(x/3)**6+1.28595570938*(x/3)**5*(y/3)-9.58232354246*(x/3)**4*(y/3)**2-17.6272057479*(x/3)**2*(y/3)**4-6.71171067242*(x/3)*(y/3)**5+0.711968481727*(y/3)**6+2.44444490116*(x/3)**7+0.492679440713*(x/3)**6*(y/3)+0.552677199957*(x/3)**5*(y/3)**2+0.549808945833*(x/3)**4*(y/3)**3+0.671044748426*(x/3)**3*(y/3)**4+0.857410587542*(x/3)**2*(y/3)**5+0.750937834496*(x/3)*(y/3)**6-0.828533309032*(y/3)**7+35.5610260815*(x/3)**8-4.47956327485*(x/3)**7*(y/3)+20.8953724365*(x/3)**6*(y/3)**2+6.83780968103*(x/3)**5*(y/3)**3+20.431214508*(x/3)**4*(y/3)**4+30.3080551192*(x/3)**3*(y/3)**5+26.3674502935*(x/3)**2*(y/3)**6+4.46487516982*(x/3)*(y/3)**7-1.44282491053*(y/3)**8-1.05931697212*(x/3)**9-0.285142290228*(x/3)**8*(y/3)-0.510256189167*(x/3)**7*(y/3)**2-0.366656219615*(x/3)**6*(y/3)**3+0.31058970607*(x/3)**5*(y/3)**4-1.28400293855*(x/3)**4*(y/3)**5-0.80943643865*(x/3)**3*(y/3)**6-0.421014759443*(x/3)**2*(y/3)**7-0.271029456555*(x/3)*(y/3)**8+0.496748239549*(y/3)**9-16.1608007815*(x/3)**10+3.00943525546*(x/3)**9*(y/3)-16.0941143617*(x/3)**8*(y/3)**2+1.09055359424*(x/3)**7*(y/3)**3-4.56595616916*(x/3)**6*(y/3)**4-22.8329695109*(x/3)**5*(y/3)**5-19.4142461099*(x/3)**4*(y/3)**6-18.6447698705*(x/3)**3*(y/3)**7-11.6454705747*(x/3)**2*(y/3)**8+0.364939418082*(x/3)*(y/3)**9
	return inv1 <= 0

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	random_seed = int(sys.argv[2])
	from itertools import count
	import time
	# print(where_inv_polySOS([-1.65, 1.8]))
	# assert False
	
	# for trained multuple actors	
	# agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=25, fc2_units=None, individual=False)
	# for individual distilled controller
	agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=50, fc2_units=None)

	scores = ddpg()
	assert False

	#random intial state test to generate the scatter plot of safe and unsafe region
	state_list, fuel_list, _ = test(agent, './models/Individual.pth', renew=True, state_list=[])
	print(len(fuel_list), np.mean(fuel_list))
	np.save('initial_state_500_poly10_err0.05.npy', np.array(state_list))

	# To compare the individual controller and Bernstein polynomial approximation controlled trajectory
	# state_list, _, indi_trajectory = test(agent, './models/Individual.pth', renew=True, state_list=[], 
	# 	EP_NUM=1, random_initial_test=False)
	# state_list, _, BP_trajectory = test(agent, './models/Individual.pth', renew=False, state_list=state_list, 
	# 	EP_NUM=1, random_initial_test=False, BP=True)
	# plt.plot(indi_trajectory[:, 0], indi_trajectory[:, 1], label='individual')
	# plt.plot(BP_trajectory[:, 0], BP_trajectory[:, 1], label='BP')
	# plt.legend()
	# plt.savefig('Control_Indi_BP.png')

	

