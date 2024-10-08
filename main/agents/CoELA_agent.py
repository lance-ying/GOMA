import copy
import math
from pathlib import Path

import numpy as np
import pdb
from utils_environment import convert_goal, check_progress, check_progress2
from LLM import *
# from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
# from maskrcnn_benchmark.config import cfg

class vision_LLM_agent:
    """
    LLM agent class
    """
    def __init__(self, agent_id, char_index, args):
        # self.vision_pipeline = None
        # self.config = vision_pipeline.agent_vision_config(
        # 	agent_type = 'LLM_vision',
        # 	char_index=char_index,
        # 	agent_id=agent_id,
        # 	)
        self.debug = args.debug
        self.agent_type = 'COELA'
        self.agent_names = ["Zero", "Alice", "Bob"]
        self.agent_id = agent_id
        self.opponent_agent_id = 3 - agent_id
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}
        self.unsatisfied = {}
        self.steps = 0
        self.location = None
        self.last_location = None
        self.plan = None
        self.stuck = 0
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None
        self.opponent_grabbed_objects = []
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id2node = {}
        self.id_inside_room = {}
        self.satisfied = []
        self.reachable_objects = []
        self.room_explored = {
            "livingroom": False,
            "kitchen": False,
            "bedroom": False,
            "bathroom": False,
        }
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.record_dir = os.path.join("record", 'image', self.agent_names[self.agent_id])
        Path(self.record_dir).mkdir(parents=True, exist_ok=True)
        self.obs = None
        self.rotated = 0


    def goexplore(self):
        target_room_id = int(self.plan.split(' ')[-1][1:-1])
        if self.rotated == 12:
            assert self.current_room['id'] == target_room_id
            self.rotated = 0
            self.plan = None
            return None
        if self.rotated > 0:
            self.rotated += 1
            return None
        if self.current_room['id'] != target_room_id or (math.dist(self.location, self.last_location) > 0.3 and self.stuck < 10): # still not at the center
            if self.debug:
                print(f"{self.agent_names[self.agent_id]}'s current location: {self.location}")
            return self.plan.replace('[goexplore]', '[walk]')
        else:
            self.rotated = 1
            return None


    def gocheck(self):
        assert len(self.grabbed_objects) < 2 # must have at least one free hands
        target_container_id = int(self.plan.split(' ')[-1][1:-1])
        target_container_name = self.plan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]
        if self.current_room['class_name'] != target_container_room:
            return f"[walk] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]
        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        if self.location == self.last_location and not target_container_id in self.id2node and f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            if target_container in self.unchecked_containers[self.current_room['class_name']]:
                self.unchecked_containers[self.current_room['class_name']].remove(target_container)
            target_container['states'].append('OPEN') # must already be opened
            if 'CLOSED' in target_container['states']: target_container['states'].remove('CLOSED')

        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        if (target_container_id in self.id2node or self.location == self.last_location) and f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan.replace('[gocheck]', '[open]')
        else:
            return self.plan.replace('[gocheck]', '[walk]')


    def gograb(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        target_object_name = self.plan.split(' ')[1]
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan = None
            return None
        assert len(self.grabbed_objects) < 2 # must have at least one free hands

        target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walk] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.plan = None
            return None
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan.replace('[gograb]', '[grab]')
        else:
            return self.plan.replace('[gograb]', '[walk]')

    def goput(self):
        # if len(self.progress['goal_location_room']) > 1: # should be ruled out
        if len(self.grabbed_objects) == 0:
            self.plan = None
            return None
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
            if self.current_room['class_name'] != target_room_name:
                return f"[walk] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            if len(self.id_inside_room[self.goal_location_id]) > 1:
                self.plan = f"[goexplore] <{target_room_name}> ({self.roomname2id[target_room_name]})"
                self.room_explored[target_room_name] = True
                return None
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        if self.current_room['class_name'] != target_room_name:
            return f"[walk] <{target_room_name}> ({self.roomname2id[target_room_name]})"

        if self.goal_location not in self.reachable_objects:
            return f"[walk] {self.goal_location}"
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and 'CLOSED' in y['states']:
                return self.plan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'
        x = self.id2node[self.grabbed_objects[0]]
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"


    def LLM_plan(self,obs_human, obs_helper, graph,id2node=None):
        if len(self.grabbed_objects) == 2:
            return f"[goput] {self.goal_location}", {}

        # print(obs_helper)

        return self.LLM.run(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id], self.room_explored, obs_human, obs_helper, self.full_graph)


    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
                    # if self.debug:
                    # 	print(satisfied)
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied


    def get_action(self, obs_helper, obs_human, goal, current_room_id, current_location, prev_messages, graph, communication=True):
        """
        :param obs: {
                'bgr': bgr_images,  # 1 * w * h * 3
                'depth': depth_images,  # 1 * w * h * 3
                'camera_info': camera_info,
                'seg_info': self.id_map[agent_id], 1 * w * h
                'room_info': self.room_info,
                'messages': self.message_said,
                'location': agent_location,  # [x, y, z]
                **self.agent_env_api(agent_id),
                **self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
                        agent_id + 1))),
            }
        :param goal:{predicate:[count, True, 2]}
        :return:
        """
        # self.vision_pipeline.deal_with_obs(obs, self.last_action)
        # symbolic_obs = self.vision_pipeline.get_graph()
        symbolic_obs = obs_helper
        subgoal = list(goal.keys())[0]
        subgoal = {subgoal: goal[subgoal]}
        print(subgoal)


        if communication:
            for i in range(len(prev_messages)):
                if prev_messages[i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i + 1]}: {prev_messages[i]}")

        self.obs = obs_helper
        # print(obs)
        self.location = current_location
        self.location[1] = 1 # fix env bug
        self.id2node = {x['id']: x for x in symbolic_obs['nodes']}
        # print(self.id2node)
        self.current_room = self.id2node[current_room_id]
        # unexplored_room = False
        # if self.unchecked_containers[self.current_room['class_name']] is None:
        # 	unexplored_room = True
        # satisfied, unsatisfied = self.check_progress(symbolic_obs, goal)

        satisfied, unsatisfied = check_progress2(symbolic_obs, subgoal)
        # print(f"satisfied: {satisfied}")
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied

        if self.debug:
            # colorids = np.stack(((ids % 10) * 10, ((ids // 10) % 10) * 10, ((ids // 100) % 10) * 10), axis=3)
            import cv2
            for i in range(len(obs['camera_info'])):
                cv2.imwrite(os.path.join(self.record_dir, f"{self.steps:03}_img.png"), np.rot90(obs['bgr'][0], axes=(0,1)))
            # print(f"symbolic_graph:\n{symbolic_obs['nodes']}\n{json.dumps(symbolic_obs['edges'], indent=4)}")

        self.grabbed_objects = []
        opponent_grabbed_objects = []
        self.reachable_objects = []
        
        for e in symbolic_obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
                opponent_grabbed_objects.append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []

        # for x in symbolic_obs['nodes']:
        #     if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
        #         for room, ungrabbed in self.ungrabbed_objects.items():
        #             if ungrabbed is None: continue
        #             j = None
        #             for i, ungrab in enumerate(ungrabbed):
        #                 if x['id'] == ungrab['id']:
        #                     j = i
        #             if j is not None:
        #                 ungrabbed.pop(j)
        #         continue
        #     self.id_inside_room[x['id']] = self.current_room['class_name']
        #     if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
        #         unchecked_containers.append(x)
            
        #     if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x['id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
        #         ungrabbed_objects.append(x)
        
        for node in symbolic_obs['nodes']:
            if node['id'] in self.grabbed_objects or node['id'] in [w['id'] for w in opponent_grabbed_objects]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    other = None
                    for i, ungrab in enumerate(ungrabbed):
                        if node['id'] == ungrab['id']:
                            other = i
                    if other is not None:
                        ungrabbed.pop(other)
                continue
            self.id_inside_room[node['id']] = self.current_room['class_name']
            if node['class_name'] in self.containers_name and 'CLOSED' in node['states'] and node['id'] != self.goal_location_id:
                unchecked_containers.append(node)
            classInUnsatisfied = node['class_name'] in set([g.split('_')[1] for g in self.unsatisfied])
            idNotInSatisfied = True
            for predicate, sat in self.satisfied.items():
                if sat == 0: continue
                for s in sat:
                    if str(node['id']) in s:
                        idNotInSatisfied = False
                        break
                if idNotInSatisfied == False:
                    break
                # if node['class_name'] == predicate.split('_')[1] and node['id'] == sat[0]['id']:
                #     idNotInSatisfied = False
                #     break
            # try:
            #     idNotInSatisfied = node['id'] not in set([y['id'] for y in self.satisfied])
            # except:
            #     pdb.set_trace()
            isGrabbable = 'GRABBABLE' in node['properties']
            notGrabbed = node['id'] not in self.grabbed_objects
            notGrabbedByOpponent = node['id'] not in set([w['id'] for w in opponent_grabbed_objects])
            if classInUnsatisfied and idNotInSatisfied and isGrabbable and notGrabbed and notGrabbedByOpponent:
                ungrabbed_objects.append(node)
            # if any([node['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([node['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in node['properties'] and node['id'] not in self.grabbed_objects and node['id'] not in [w['id'] for w in opponent_grabbed_objects]:
            #     ungrabbed_objects.append(node)

        if self.room_explored[self.current_room['class_name']] and type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': symbolic_obs,
                "obs": {
                    "location": self.location,
                    "grabbed_objects": self.grabbed_objects,
                    "opponent_grabbed_objects": self.opponent_grabbed_objects,
                    "reachable_objects": self.reachable_objects,
                    "progress": {
                        "unchecked_containers": self.unchecked_containers,
                        "ungrabbed_objects": self.ungrabbed_objects,
                        },
                    "satisfied": self.satisfied,
                    "goal_position_room": self.id_inside_room[self.goal_location_id],
                    "with_character_id": self.agent_id,
                    "current_room": self.current_room['class_name'],
                    "see_this_step": list(self.id2node.keys()),
                    }
                }
        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects
        action = None
        message = None
        LM_times = 0
        # if unexplored_room:
        # 	self.action_history.pop()
        # 	self.plan = f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"
        # 	self.action_history.append(self.plan)
        # 	self.last_location = [0, 0, 0]
        # while action is None:
        if self.plan is None:
            if LM_times > 0:
                print(info)

            output = self.LLM_plan(obs_human, obs_helper, self.full_graph)

            if len(output) == 3:

                plan, a_info, message = output

            else:
                plan, a_info = output

            print("plan", plan)
            if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                self.room_explored = {
                    "livingroom": False,
                    "kitchen": False,
                    "bedroom": False,
                    "bathroom": False,
                }
                plan = f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"
            self.plan = plan
            if plan.startswith('[goexplore]'):
                self.room_explored[plan.split(' ')[1][1:-1]] = True
            self.last_location = [0, 0, 0]
            a_info.update({"steps": self.steps})
            info.update({"LLM": a_info})
        if self.plan.startswith('[goexplore]'):
            action = self.goexplore()
        elif self.plan.startswith('[gocheck]'):
            action = self.gocheck()
        elif self.plan.startswith('[gograb]'):
            action = self.gograb()
        elif self.plan.startswith('[goput]'):
            action = self.goput()
        else:
            raise ValueError(f"unavailable plan {self.plan}")

        self.steps += 1
        info.update({"plan": self.plan,
                     })
        if action == self.last_action and self.current_room['class_name'] == self.last_room:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_action = action
        self.last_location = self.location
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.plan = None
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walk] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walk] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0
        return action, info, message


    def reset(self, obs, full_graph, goal_objects_name, goal):
        # self.vision_pipeline = vision_pipeline.Vision_Pipeline(self.config, obs)
        self.steps = 0
        self.containers_name = [
			'bathroomcabinet',
			'kitchencabinet',
			'cabinet',
			'fridge',
			'stove',
			# 'coffeepot',
			'dishwasher',
			'microwave']

        # print(full_graph, goal)
        # pdb.set_trace()
        self.goal_objects_name = goal_objects_name
        self.full_graph = full_graph
        self.rooms_name = ["livingroom", "kitchen", "bedroom", "bathroom"]
        self.roomname2id = {x['class_name']: x['id'] for x in full_graph['nodes'] if x['category'] == 'Rooms'}
        # self.roomname2id = {x['class_name']: x['id'] for x in obs['room_info']}
        self.id2node = {x['id']: x for x in full_graph['nodes']}

        
        self.stuck = 0
        self.last_room = None

        subgoal =list(goal.keys())[0]
        subgoal = {subgoal: goal[subgoal]}

        satisfied, unsatisfied = check_progress2(
            obs, subgoal
        )
        
        self.unsatisfied = unsatisfied
        self.satisfied = satisfied
        self.goal_location = list(goal.keys())[0].split('_')[-1]
        self.goal_location_id = int(self.goal_location)
        try:
            self.goal_location = f"<{self.id2node[self.goal_location_id]['class_name']}> ({self.goal_location_id})"
        except:
            pdb.set_trace()

        self.id_inside_room = {self.goal_location_id: self.rooms_name[:], self.opponent_agent_id: None}
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.opponent_grabbed_objects = []

        self.room_explored = {
            "livingroom": False,
            "kitchen": False,
            "bedroom": False,
            "bathroom": False,
        }
        self.location = self.id2node[self.agent_id]['obj_transform']['position']
        self.last_location = [0, 0, 0]
        self.last_action = None
        self.rotated = 0

        # print(obs)

        for edge in obs['edges']:
            if edge['from_id'] == 2 and edge['relation_type'] == 'INSIDE':
                self.current_room = self.id2node[edge['to_id']]
                break
        self.id2node = {}
        self.plan = None
        self.action_history = [f"[goto] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []
        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.goal_location_id, self.unsatisfied)