import gym
import numpy as np
import matplotlib.pyplot as plt
import time

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    counter = 0
    rewards = []
    for _ in range(200):
        counter += 1
        mult_param_observ = np.matmul(parameters,observation)
        action = 0 if mult_param_observ < 0 else 1
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        #print("observation:", observation, "mult_param_observ:", mult_param_observ, "reward:", reward)
        totalreward += reward
        # To be able to render in macOS, see https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
        # Create a file ~/.matplotlib/matplotlibrc there and add the following code: backend: TkAgg
        #env.render()
        #time.sleep(.1)
        if done:
            break
    return totalreward, counter, rewards

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-experiments/', force=True)

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in range(10000):
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        print("parameters: ", parameters)
        reward, episode_counter, rewards = run_episode(env,parameters)
        print("rewards: ", rewards)
        print("episode: ", counter, " -> you've reached ", reward)

        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                env.close()
                break
        
    if submit:
        for _ in range(100):
            run_episode(env,bestparams)
        env.monitor.close()

    return counter

# train an agent to submit to openai gym
# train(submit=True)

# create graphs
results = []
for _ in range(1000):
    results.append(train(submit=False))

print(results)
print("results mean:", np.mean(results))

plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()


print(np.sum(results) / 1000.0)