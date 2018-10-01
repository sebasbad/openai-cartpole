import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    counter = 0
    for _ in range(200):
        #env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        counter += 1
        if done:
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-hill/', force=True)

    episodes_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 - 1
    bestreward = 0
    counter = 0

    for _ in range(2000):
    #for _ in range(100):
        counter += 1
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        #print(newparams)
        rewards = []
        reward = 0
        for _ in range(episodes_per_update):
            run = run_episode(env,newparams)
            reward += run
            rewards.append(run)
        #reward = run_episode(env,newparams)
        #print("counter %d reward %d best %d" % (counter, reward, bestreward))
        best_in_5_episodes = max(rewards)
        #print("global_counter %d counter %d best_in_5_episodes %d best %d" % (global_counter, counter, best_in_5_episodes, bestreward))
        #print("rewards %a best %d" % (rewards, best_in_5_episodes))
        if reward > bestreward:
        #if best_in_5_episodes > bestreward:
            #print("update")
            bestreward = reward
            #bestreward = best_in_5_episodes
            parameters = newparams
            if reward == 200:
                break

    if submit:
        for _ in range(100):
            run_episode(env,parameters)
        env.monitor.close()
    return counter


# train an agent to submit to openai gym
# train(submit=True)

import datetime

# create graphs
results = []
global_counter = 0
for _ in range(1000):
    global_counter += 1
    print("global_counter %d" % (global_counter))
    print(datetime.datetime.now())
    result = train(submit=False)
    print(datetime.datetime.now())
    print("global_counter %d result %d" % (global_counter, result))
    results.append(result)

print(results)
print("results mean:", np.sum(results) / 1000.0)

plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Hill Search')
plt.show()
