from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.Team40Agent import Team40Agent, LiarAgent, LazyAgent
from agents1.BW4THuman import Human


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name': 'agent1', 'botclass': Team40Agent, 'settings': {'do_log': True}},
        {'name': 'agent2', 'botclass': LazyAgent, 'settings': {}},
        {'name': 'agent3', 'botclass': LiarAgent, 'settings': {}},
        {'name': 'agent4', 'botclass': Team40Agent, 'settings': {}}
        #{'name': 'agent5', 'botclass': Team40Agent, 'settings': {'do_log': True}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
