import os
import gym
from sac_agent import SAC_Agent
import numpy as np    
import matplotlib.pyplot as plt


def scale_action(env, action):
    slope = (env.action_space.high - env.action_space.low) / 2
    action = env.action_space.low + slope * (action + 1)
    return action

def run(env, agent, num_episodes, max_steps, train, render=False, exploration_episodes=0):
    stats = {'episode_rewards': [], 'episode_lengths': []}
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            if episode < exploration_episodes:
                action = np.random.uniform(-1, 1, env.action_space.shape[0])
            else:
                action = agent.act(state, deterministic = not train) 

            env_action = scale_action(env, action)
            next_state, reward, done, info = env.step(env_action)

            if(train):
                agent.update(state, action, next_state, reward, done)

            if(render):
                env.render()

            state = next_state
            episode_reward += reward

            if(done):
                break

        #End of episode
        print("Episode:", str(episode + 1), "- Reward:", episode_reward, "- Steps:", step)
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(step)
        if train and (episode % 10 == 0 or episode == num_episodes-1):
            agent.save("models/sac_model_" + str(episode) + ".pth")
    env.close()
    return stats


env = gym.make('BipedalWalker-v3')
max_steps = 1000
agent = SAC_Agent(env)

#num_episodes = 1000
#train = True
#stats = run(env, agent, num_episodes, max_steps, train)
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#plt.plot(stats['episode_rewards'])
#plt.show()

agent.load("models/sac_model_500.pth")
num_episodes = 5
train = False
stats = run(env, agent, num_episodes, max_steps, train, render=True)
print(np.mean(stats['episode_rewards']), np.std(stats['episode_rewards']))