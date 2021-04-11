from src.DQAgent import DQAgent
from src.QAgent import QAgent

# epsilon = 0.1
# alpha = 0.5
# gamma = 0.8
# agent = QAgent(alpha, gamma, epsilon)
# agent.train(500, 'test4')
# agent.test_one()

epsilon = 0.7
alpha = 0.0005
gamma = 0.85
agent = DQAgent(alpha, gamma, epsilon)
score, height = agent.train(200)
print("Best training score: {}, height: {}".format(score, height))
agent.save_model('test1')
agent.test_one()

