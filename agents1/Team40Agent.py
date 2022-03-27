
from typing import final, List, Dict, Final
import enum, random, copy
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction, CloseDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

class Phase(enum.Enum):
    PLAN_PATH_TO_ROOM = 1,
    FOLLOW_PATH_TO_ROOM = 2,
    OPEN_DOOR = 3,
    DECIDE_ACTION = 4,
    PLAN_PATH_TO_DROP_OBJECT = 5,
    FOLLOW_PATH_TO_DROP_OBJECT = 6,
    DROP_OBJECT = 7,
    INITIATE_ROOM_SEARCH = 8,
    SEARCH_ROOM = 9,
    FOUND_BLOCK = 10,
    EXIT_ROOM = 11

class Team40Agent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._isFirstAction = True
        self._phase = Phase.DECIDE_ACTION
        self._teamMembers = []
        self._activeObjectives = []
        self._state_tracker = None
        self._navigator = None

        self._msgHist = {}
        self._oldMsg = {}
        self._latestMsg = {}
        self._trustPerMember = {}

        self._memberObjects = {}

        self._doNothing = False
        if 'do_nothing' in settings and settings['do_nothing']:
            self._doNothing = True

        self._doLog = False
        if 'do_log' in settings and settings['do_log']:
            self._doLog = True

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                    algorithm=Navigator.A_STAR_ALGORITHM)

    def _removePrefix(self, pre, message):
        if message.startswith(pre):
            return message[len(pre):]

    def _parseLocation(self, message):
        ind = message.index('at location')
        return message[ind + len('at location '):]

    def _parseBlockVisual(self, message):
        ind = message.index('{')
        tempMsg = message[ind:]
        for charInd in range(len(tempMsg)):
            if tempMsg[charInd] == '}':
                return tempMsg[:charInd + 1]

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        self._oldMsg = copy.deepcopy(self._latestMsg)

        for member in teamMembers:
            for mssg in self.received_messages:
                if mssg.from_id == member:
                    self._latestMsg[member] = mssg.content

                    if mssg.content not in self._msgHist:
                        self._msgHist[member].append(mssg.content)

            if member not in self._oldMsg:
                self._oldMsg[member] = None
            if member not in self._latestMsg:
                self._latestMsg[member] = None
            if member not in self._msgHist:
                self._msgHist[member] = []

    def _updateTrusts(self, newestMsg):
        for member in newestMsg.keys():
            # Decrease member trust by default
            self._updateTrustBy(member, -0.002)
            if self._trustPerMember[member] < 0:
                self._trustPerMember[member] = 0

            message = newestMsg[member]
            if message is None:
                continue

            # Colorblind agent
            if ('Found' in message or 'Picking' in message or 'Dropped' in message) and 'colour' not in message:
                self._updateTrustBy(member, -0.1)
                continue

            # Liar agent - picking/dropping objects
            if 'Picking up goal block {' in message:
                self._memberObjects[member].append(self._parseBlockVisual(message))
            if len(self._memberObjects[member]) > 2:
                self._updateTrustBy(member, -0.2)
                continue
            if 'Dropped goal block {' in message:
                if len(self._memberObjects[member]) <= 0:
                    self._updateTrustBy(member, -0.2)
                    continue
                vis = self._parseBlockVisual(message)
                found = False
                for indObj in range(len(self._memberObjects[member])):
                    if self._memberObjects[member][indObj] == vis:
                        self._memberObjects[member].pop(indObj)
                        found = True
                if not found:
                    self._updateTrustBy(member, -0.2)
                    continue

            # Liar agent - detect messages in impossible order
            if 'Opening door of' in message and \
                    ('Moving to ' + self._removePrefix('Opening door of ', message)) not in self._msgHist[member]:
                self._print(member + ' lied')
                self._updateTrustBy(member, -0.2)
                continue
            if 'Searching through' in message and \
                    ('Moving to ' + self._removePrefix('Searching through ', message)) not in self._msgHist[member]:
                self._print(member + ' lied')
                self._updateTrustBy(member, -0.2)
                continue
            if 'Found goal' in message and \
                    ('Searching through ' + self._parseLocation(message)) not in self._msgHist[member]:
                self._print(member + ' lied')
                self._updateTrustBy(member, -0.2)
                continue
            if 'Picking' in message and \
                    ('Moving' not in self._msgHist[member] or 'Searching' not in self._msgHist[member]):
                self._print(member + ' lied')
                self._updateTrustBy(member, -0.2)
                continue
            if 'Dropped' in message and 'Picking' not in self._msgHist[member]:
                self._print(member + ' lied')
                self._updateTrustBy(member, -0.2)
                continue

            # Check order of messages
            oldMsg = self._oldMsg[member]
            if oldMsg is not None and oldMsg != message:

                # Lazy agent - not searching room after opening
                if 'Opening' in oldMsg and 'Searching' not in message:
                    self._updateTrustBy(member, -0.1)
                    continue
                # Lazy agent - not picking up although not carrying anything
                if 'Found' in oldMsg and 'Picking' not in message:
                    if 'Picking' not in self._msgHist[member]:
                        self._updateTrustBy(member, -0.1)
                        continue

                # Normal cases - Check if room is consistent
                if 'Moving to' in oldMsg and 'Opening door of' in message:
                    if self._removePrefix('Moving to ', oldMsg) == self._removePrefix('Opening door of ', message):
                        self._updateTrustBy(member, 0.1)
                        continue
                if 'Opening door of' in oldMsg and 'Searching through' in message:
                    if self._removePrefix('Opening door of ', oldMsg) == \
                            self._removePrefix('Searching through ', message):
                        self._updateTrustBy(member, 0.1)
                        continue
                if 'Found goal block' in oldMsg and 'Picking up goal block' in message:
                    if self._parseLocation(oldMsg) == self._parseLocation(message):
                        if self._parseBlockVisual(oldMsg) == self._parseBlockVisual(message):
                            self._updateTrustBy(member, 0.2)
                            continue
                if 'Picking up goal block' in oldMsg and 'Dropped goal block' in message:
                    if self._parseBlockVisual(oldMsg) == self._parseBlockVisual(message):
                        self._updateTrustBy(member, 0.2)
                if 'Dropped goal block' in oldMsg and 'Moving to' in message:
                    self._updateTrustBy(member, 0.1)

    def _updateTrustBy(self, member, amount):
        current = self._trustPerMember[member]
        if amount == 0:
            return
        elif current + amount > 1:
            self._trustPerMember[member] = 1
        elif current + amount < 0:
            self._trustPerMember[member] = 0
        else:
            self._trustPerMember[member] += amount

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']

        # Get goal objectives initially
        if self._isFirstAction:
            # Add team members
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)

            for member in self._teamMembers:
                # Default trust values to 0.5
                self._trustPerMember[member] = 0.5

                # Initialize object list per member
                self._memberObjects[member] = []

            self._activeObjectives = [goal for goal in state.values()
                                      if 'is_goal_block' in goal and goal['is_goal_block']]
        self._isFirstAction = False

        # Process messages from team members
        self._processMessages(self._teamMembers)
        newMessages = {}
        for member in self._teamMembers:
            if self._latestMsg[member] == self._oldMsg[member]:
                newMessages[member] = None
            else:
                newMessages[member] = self._latestMsg[member]
        # Update member trusts based on newest messages
        self._updateTrusts(newMessages)

        if self._doNothing:
            print(self._memberObjects)
            return None, {}

        while True:
            if Phase.DECIDE_ACTION == self._phase:
                self._carrying = state[self.agent_id]['is_carrying']

                # Carrying something, go drop it
                if len(self._carrying) != 0:
                    ind = self._indexObjEquals(self._activeObjectives, self._carrying[0])

                    # Holding a useless block
                    if ind == -1:
                        return None, {}
                    else:
                        self._phase = Phase.PLAN_PATH_TO_DROP_OBJECT

                # Still have to look for goal objects
                elif len(self._activeObjectives) != 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM

                # No job left to do
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
                self._sendMessage('Dropped goal block ' + str(self._carrying[0]['visualization']) +
                                  ' at location ' + str(self._loc_goal), agent_name)
                self._activeObjectives.pop(self._indexObjEquals(self._activeObjectives, self._carrying[0]))
                return DropObject.__name__, {'object_id': self._carrying[0]['obj_id']}

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values() if 'class_inheritance' in door
                               and 'Door' in door['class_inheritance'] and not door['is_open']]
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
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
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
                waypoints = [(doorLoc[0], doorLoc[1]-1), (doorLoc[0], doorLoc[1]-2), (doorLoc[0]-1, doorLoc[1]-2), (doorLoc[0]-1, doorLoc[1]-1)]
                self._navigator.add_waypoints(waypoints)
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._roomIsUseless = True
                self._phase = Phase.SEARCH_ROOM

            if Phase.SEARCH_ROOM == self._phase:
                self._state_tracker.update(state)

                nearby_objects = [obj for obj in state.values() if 'is_collectable' in obj and obj['is_collectable']]
                objInd = self._indexObjEquals(nearby_objects, self._activeObjectives[0])

                if self._hasCommon(nearby_objects, self._activeObjectives):
                    self._roomIsUseless = False

                if objInd != -1:
                    self._navigator.reset_full()
                    self._searched_obj = nearby_objects[objInd]
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

                if not self._roomIsUseless:
                    return CloseDoorAction.__name__, {'object_id': self._door['obj_id']}

    def _indexObjEquals(self, objList, obj):
        for i in range(len(objList)):
            if self._objEquals(objList[i], obj):
                return i
        return -1

    def _indexObjsEquals(self, objList1, objList2):
        for i in range(len(objList1)):
            for j in range(len(objList2)):
                if self._objEquals(objList1[i], objList2[j]):
                    return i, j
        return -1, -1

    def _hasCommon(self, li1, li2):
        for obj1 in li1:
            for obj2 in li2:
                if self._objEquals(obj1, obj2):
                    return True
        return False

    def _objEquals(self, obj1, obj2):
        ov1 = obj1['visualization']
        ov2 = obj2['visualization']

        if ov1['shape'] != ov2['shape']:
            return False
        if ov1['colour'] != ov2['colour']:
            return False
        return True

    def _print(self, msg):
        if self._doLog:
            print(msg)
