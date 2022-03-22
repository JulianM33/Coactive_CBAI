
from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction, CloseDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

# Procedure
# 1. check objectives
#       - announce objectives upon reaching them
#       - if a highly trustworthy agent announces, cancel route and add objectives to active list
# 2. plan room to check
#       - remove from interest the destination of a trustworthy agent's destination
#       - announce which room to check
# 3. check room
#       - if found block -> step 4
#       - if found multiple blocks -> announce
#       - if no block -> step 2
# 4. drop block
#       -


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    PLAN_PATH_TO_OBJECTIVE1 = 4,
    PLAN_PATH_TO_OBJECTIVE2 = 5,
    PLAN_PATH_TO_OBJECTIVE3 = 6,
    DECIDE_ACTION = 7,
    PLAN_PATH_TO_DROP_OBJECT = 8,
    FOLLOW_PATH_TO_DROP_OBJECT = 9,
    DROP_OBJECT = 10,
    INITIATE_ROOM_SEARCH = 11,
    SEARCH_ROOM = 12,
    FOUND_BLOCK = 13,
    EXIT_ROOM = 14,
    CLOSE_DOOR = 15

class Team40Agent(BW4TBrain):

    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)
        self._isFirstAction = True
        self._phase = Phase.DECIDE_ACTION
        self._teamMembers = []
        self._activeObjectives = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state:State):
        agent_name = state[self.agent_id]['obj_id']

        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)

        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        # get goal objectives initially
        if self._isFirstAction:
            self._activeObjectives = [goal for goal in state.values()
                                      if 'is_goal_block' in goal and goal['is_goal_block']]

        self._isFirstAction = False

        while True:
            if Phase.DECIDE_ACTION == self._phase:
                self._carrying = state[self.agent_id]['is_carrying']

                # carrying something, go drop it
                if len(self._carrying) != 0:
                    ind = self._indexObjEquals(self._activeObjectives, self._carrying[0])

                    # holding a useless block
                    if ind == -1:
                        return None, {}
                    else:
                        self._phase = Phase.PLAN_PATH_TO_DROP_OBJECT

                # still have to look for goal objects
                elif len(self._activeObjectives) != 0:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

                # no job left to do
                else:
                    return None, {}

            if Phase.PLAN_PATH_TO_DROP_OBJECT == self._phase:
                self._navigator.reset_full()

                ind = self._indexObjEquals(self._activeObjectives, self._carrying[0])
                self._loc_goal = self._activeObjectives[ind]['location']

                self._navigator.add_waypoint(self._loc_goal)
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OBJECT

            if Phase.FOLLOW_PATH_TO_DROP_OBJECT == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                self._phase = Phase.DECIDE_ACTION
                self._sendMessage('Dropped goal block ' + str(self._carrying[0]['visualization'])
                                  + ' at location ' + str(self._loc_goal),
                                  agent_name)
                self._activeObjectives.pop(0)
                return DropObject.__name__, {'object_id': self._carrying[0]['obj_id']}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door
                               and 'Door' in door['class_inheritance']
                               and not door['is_open']]
                if len(closedDoors) == 0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1]+1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}   
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.INITIATE_ROOM_SEARCH
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.INITIATE_ROOM_SEARCH == self._phase:
                self._navigator.reset_full()
                doorLoc = self._door['location']
                waypoints = [(doorLoc[0], doorLoc[1]-1),
                             (doorLoc[0], doorLoc[1]-2),
                             (doorLoc[0]-1, doorLoc[1]-2),
                             (doorLoc[0]-1, doorLoc[1]-1)]
                self._navigator.add_waypoints(waypoints)
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._roomIsEmpty = True
                self._phase = Phase.SEARCH_ROOM

            if Phase.SEARCH_ROOM == self._phase:
                self._state_tracker.update(state)

                nearby_objects = [obj for obj in state.values()
                                  if 'is_collectable' in obj and obj['is_collectable']]
                if len(nearby_objects) != 0:
                    self._roomIsEmpty = False
                nby_obj_ind = self._indexObjEquals(nearby_objects, self._activeObjectives[0])
                if nby_obj_ind != -1:
                    self._navigator.reset_full()
                    self._searched_obj = nearby_objects[nby_obj_ind]
                    self._navigator.add_waypoint(self._searched_obj['location'])
                    self._phase = Phase.FOUND_BLOCK
                    self._sendMessage('Found goal block ' + str(self._searched_obj['visualization'])
                                      + ' at location ' + str(self._searched_obj['location']),
                                      agent_name)
                else:
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action is not None:
                        return action, {}
                    self._phase = Phase.EXIT_ROOM

            if Phase.FOUND_BLOCK == self._phase:
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.EXIT_ROOM
                self._sendMessage('Picking up goal block ' + str(self._searched_obj['visualization'])
                                  + ' at location ' + str(self._searched_obj['location']),
                                  agent_name)
                return GrabObject.__name__, {'object_id': self._searched_obj['obj_id']}

            if Phase.EXIT_ROOM == self._phase:
                self._navigator.reset_full()
                doorLoc = self._door['location']
                self._navigator.add_waypoint((doorLoc[0], doorLoc[1] + 1))
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.DECIDE_ACTION

                if not self._roomIsEmpty:
                    return CloseDoorAction.__name__, {'object_id': self._door['obj_id']}

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)       
        return receivedMessages

    def _trustBlief(self, member, received):
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs

    def _indexObjEquals(self, objList, obj):
        for i in range(len(objList)):

            if self._objEquals(objList[i], obj):
                return i
        return -1

    def _objEquals(self, obj1, obj2):
        ov1 = obj1['visualization']
        ov2 = obj2['visualization']

        if ov1['shape'] != ov2['shape']:
            return False
        if ov1['colour'] != ov2['colour']:
            return False
        return True
