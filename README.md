GOMA: Proactive Embodied Cooperative Communication via Goal-Oriented Mental Alignment

The GOMA algorithm casts human robot communication as a planning problem by selecting utterances that maximizally improves the efficiency of the joint plan in a partially observable environment.

Reward of robot sharing information X to human:
R(request X) = KL(E[human plan | human mind + X] || E[human plan | human mind ]) - C 

Reward of robot requesting information X from human:
R(request X) = KL(E[robot plan | robot mind + X] || E[robot plan | robot mind ]) - C 

where C is the communication cost.


The current implementation is tested in the virtual home domain. For installing virtual home, please clone this repo: https://github.com/xavierpuigf/virtualhome
