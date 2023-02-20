# gym-flp
Implements different discrete and continuous Facility Layout Problem representations

## Purpose
This package aims at supporting Facility Layout Problem (FLP) and/or Operations Research (OR) and/or Reinforcement Learning (RL) researchers/practitioners in solving FLPs. FLP/OR researchers/practicioners can implement their problems into the package and leverage the Gym backbone to quickly start training RL algorithms on their problems. We therefore provide them with a sort of benchmark environment repository which are common in RL research. RL researchers, in turn, may benefit from a hands-on, real-life manufacturing problem set to test RL algorithm advances on beyond toy problems.

## Installation
gym-flp can be installed via PyPi:
```
pip install gym-flp
```

Alternative clone this repo and install locally using
```
pip install gym-flp -e .
```

Please note that the algorithm files may require additional package not covered by setup.py, such as Stable-baseline3, imageio, rich, tqdm, matplotlib, torch, tensorboard.
## Usage
Any environment must be instantiated with the `gym.make()` method by passing the environment's id, like \verb|'cartpole-v0'|, \verb|'breakout-v0'| or \verb|'taxi-v1'|.

Step-size based environments (those using Discrete / Multi-Discrete action spaces are purposefully designed to be at risk of moving facilities beyond the plant boundaries. With this violating feasibility restrictions, these actions must incur negative rewards.

Hence, by using `Spaces` we can conveniently leverage the in-built features of Gym, by penalizing any action that leads to off-grid positioning or violations of spatial requirements like minimum side lengths assigning extraordinarily high negative rewards to all actions which lead to states for which the method \verb|env.observation_space.contains(state)| returns \verb|False|.)

### Implementation Details
Furthermore, every environment must have at least these three methods:
+ reset() resets the environment to a pre-defined or random state and returns
an observation
+ step(action) performs the action passed to the method, steps the environment
by one timestep, and returns the consequences observation, reward,
done and info which are explained in more detail below.
+ render() prints one frame of the environment.

#### Supported environment variants

1) Quadratic Assignment Problem

The Quadratic Assignment Problem (QAP) goes back to Koopmans and Beckmann [30]. Here, the plant site is divided into rectangular blocks with the same
area and shape, and one facility is assigned to each block iteratively until an
optimal assignment is identified in terms of low material handling costs [17].
That being said, as the true content of a block is irrelevant for the solution,
the QAP can also be used to solve problems outside of a plant context, such
as facility location problems. Our environment design follows the original formulation of the QAP by Koopmans and Beckmann [30]:
Given is a set of facilities N = {1, 2, ...n} to be assigned to a set of locations
of the same size N. A permutation p represents the non-repeating unique
assignment of facilities 1 though n to locations. Thus, Sn = p : N → N is the
set of all permutations. Further given are the flow matrix F = (fij ) as an nxn
matrix where fij represents a flow between the facilities i and j as well as the
distance matrix D = (dij ) as an nxn matrix where dij is the fixed distance
between the locations i and j. The goal of a QAP is to solve the optimization
problem

minp∈Sn Xn i=1 Xn j=1 fij ∗ dp(i)p(j)

where each product fij∗dp(i)p(j)
is the cost of assigning facility i to location p(i)
and facility j to location p(j). Some formulations, e.g. in [29] or [47], include
the term cij in the product to account for different cost per unit flows. The
implementation in this package, however, assumes these cost to be uniform.
In this package, QAP observations are represented as a permutation vector
with n integer values which means the observation space coincides with the
permutation set Sn. The actions space is built using a pairwise exchange logic
and thus consists of n − n ∗ 0.5 + 1 discrete actions obtained as follows: every
facility i can be swapped with another facility j leading to n options. The
action space can be halved by the fact that the swap i → j is identical to
j → i. The swaps for i = j make no change to the permutation allowing us
to omit these. However, we retain one ’idle’ action to permit the agent to
stay in the state it deems optimal. This procedure is somewhat inspired by
OpenAI’s Rubik’s cube solving robot [42] as this motivated the authors to
include an image representation of the observations, too. In a similar fashion
how the robot looks for the desired pattern (all faces of the cube have only one
colour) and learns to find the fastest trajectory of manipulations to achieve
this, agents using the image mode of gym-flp should be able to detect which
actions lead to a favourable permutation. Details for this idea are given in
the section ’Observation Spaces’. The QAP based environment for discrete
problems can be invoked by passing the id qap-v0 to env.make(id). qap-v0
takes the optimal arguments mode (allowed values: ’human’ and ’rgb array’)
and instance.

2) Flexible Bay Structure
The Flexible Bay Structure (FBS) notation has been created by Tong [54]. It
allows the departments to be located only in parallel bays with varying widths.
Bays are bounded by straight aisles on both sides, and departments are not
allowed to span over multiple bays [29]. The width of the bays is determined
by the cumulated area demand of the facilities assigned to each one using the
equation below [23]:
Length of bayi =
Total department areas in bayi
Width of the facility (2)
Following the convention, the FBS notation moves from top to bottom
and left to right, locating the point of origin in the top-left corner. Conveniently, this relates well to the indices in 2-dimensional numpy arrays although
in flipped order (rows = axis 0 and columns = axis 1). To facilitate the conversion of array data to an image for further use with Deep Learning-based
approaches, we use this convention throughout all continuous formulations. All
variables named ”width” refer to y coordinates (i.e. rows in an array) and all
variables ”length” refer to x coordinates (columns), respectively. W and L are
the overall plant dimensions, whereas w and l are the dimensions of individual
facilities. Fig. 1 below summarizes the terminology used to describe the states
in continuous problems.
Layouts are generated by manipulating two vectors: the permutation p,
indicating the order in which facilities are located, and the bay breaks b. This
is a binary vector that represents the last facility in a bay and is of the same
length as the permutation. The interpretation of this vector varies throughout
literature: while Garcia-Hernandez et al. indicate a bay division with the value
1 [20], Ulutas and Konak represent break positions with a 0 [23]. We choose
to go by the former interpretation. This means that the last position in b
is always equal to 1. For instance, in Fig. 1, the bay break vector would be
b = {0, 0, 1, 0, 0, 1, 0, 0, 1}. For clarity, the positions in the permutation and
the bay breaks they match are highlighted in bold in the figure.
Both p and b are latent variables that cannot be observed but inferred by
the agent. Instead, FBS observations are provided either as a vector that holds
the centre coordinates in x and y, the length li and the width wi of all facilities
in a problem or as an RGB image representation of this information, depending
on the mode selected. For the action space, we implemented the four actions
”Randomize”, ”Bit Swap”, ”Bay Exchange”, and ”Inverse” according to [20]
plus the common ’idle’ action as in the QAP environment before.
The FBS environment for continuous problems can be invoked by passing
the id fbs-v0 to env.make(id). It consumes the following optional arguments:
mode (allowed values: ’human’ and ’rgb array’) and instance.

3) Slicing Tree Structure
The Slicing Tree Structure (STS) goes back to [53]. In line with the FBS
implementation, we use the same spatial variables to describe STS layouts,
i.e. W, L, w, and l. However, as does FBS, STS usually assumes an overall
plant area that is equal to the sum of facility area requirements.

Following [19] and [45], we use a layout encoding that encompasses three
distinct items: a permutation vector p, a slicing order vector s, and an orientation vector o. While p is of size i = n, o and s are of size j = n−1. p has the
same meaning as in QAP and FBS. s describes the position of a cut within
the layout, and o denotes the direction of the cut. This package uses a 0 to
indicate vertical cuts and a 1 for horizontal cuts, respectively. This is the key
difference in comparison with FBS, as STS allows cuts in two rather than one
direction.
Given the plant dimensions W, L and all area requirements ai for the facilities, the layout creation follows the following procedure [46]:
1. Build a slicing tree by sequentially cutting p at the positions indicated by
s in direction o and assign the thus created new vectors to the sub-layouts.
For any vector slice of size 1, i.e. holding only one facility, it is assigned as
outer node (tree leaf).
2. Starting from the leaves in the lowest tree level, the area requirement of
their parent node is computed. This is repeated in a bottom-up manner
until the root node is reached.
3. The empty layout will now be sliced along the order as given by s. The
slicing position is determined similar to Eq. (1) above, i.e. by dividing
the area required in one sub-layout by the width or length of the plant
(depending on the slicing direction).
Fig. 2 shows an illustrative example of the STS approach. The layout in Fig.
2b is encoded by the permutation p = {4, 2, 1, 6, 7, 8, 5, 3, 9}, the slicing order
s = {4, 5, 3, 2, 8, 1, 6, 7} and the slicing orientations o = {0, 1, 1, 0, 0, 1, 1, 0}
leading to the slicing tree in 2a. After the first cut, located in the left-hand
child node after the root, the layout is split into the sub-layouts with the
permutation slices p1,l = {4, 2, 1, 6} and p1,r = {7, 8, 5, 3, 9}. The index of the
superscript indicates the number of cuts performed in the range 1...n − 1, and
the letter indicates the side of the new sub-layout. Given a vertical cut, the
width of the sub-layout corresponds to W whereas the length can be computed
as l1,l = (a4 + a2 + a1 + a6)/W. This computation is repeated recursively
until the leaf nodes are reached. For a more detailed explanation of the STS
approach, we refer to Friedrich et al. [19]
Like in the FBS environment, STS observations are provided as a vector
with centre-point coordinates in xi and yi
, the length li and the width wi of the
facilities, or their respective RGB image representation in rgb array mode. For
the action space, we implemented the five actions ”Permute”, ”Slice Swap”,
”Bit Swap”, ”Shuffle” and ”Idle”.
The STS environment for continuous problems can be invoked by passing
the id sts-v0 to env.make(id). sts-v0 accepts the arguments mode (allowed
values: ’human’ and ’rgb array’) and instance.

4) Open Field Layout Problem
The open field problem (OFP) is based on the collision-free unequal area FLP
mixed-integer linear problem (MILP) solution by Montreuil et al. [40]. From
a practice-based factory planning perspective this is closest resemblance of a
real-world planning approach. The OFP environment is characterised by the
same size variables as the other continuous environments above, i.e. W, L, w
and l. This approach does not rely on divisions being made, but facilities
can move around freely in the plant, constrained by the plant dimensions. To
build layouts, no encoding mechanisms are required other than the facility
dimensions or areas and their coordinates. While this environment is inspired
by collision-free programming approaches like in [40], we opted for the name
open field problem for two reasons: First, contrary to MIP approaches, we
allow floating-point values for coordinates and side lengths (if the problem set
considered contains area requirements rather than dimensions). Second, by
passing True to the argument greenfield, plant dimensions are disregarded
in favour of a greenfield approach. In this case, W and L are programmatically
set to be five times the size compared to the area demands.
As with the former continuous formulations, observations in the OFP environment are given as a vector with centre-point coordinates x and y, length
li and width wi of the facilities or an RGB image. The action space is set up
as a function of the problem size, i.e. 5 ∗ n + 1. This means that on top of the
common ’idle’ action introduced before, every facility can be moved with 5
actions: up, down, left, right, and rotate. Note that, due to the representation
of layouts in a numpy array, a facility moving ”up” in the layout will actually
move down in terms of array row indices.
The OFP environment allows us to best leverage the in-built features of
Gym by penalizing any action that leads to off-grid positioning or violations
of spatial requirements like minimum side lengths by assigning extraordinarily high negative rewards to all states which the method env.observation_
space.contains(env.state) evaluates to False. Plus, we can conveniently
penalize actions leading to collisions.
The greenfield environment for discrete problems can be invoked by passing the id ofp-v0 to env.make(id). ofp-v0 further takes these optional arguments: mode (allowed values: ’human’ and ’rgb array’), instance, aspect_
ratio (floating point), step_size (integer, indicating the number of distance
units a facility is moved) and greenfield (boolean, indicating whether or not
the provided plant dimensions should be used).

#### Supported Problem instances

| Problem set | Flow source | Facility dimensions source | Plant sizes included |
|-------------|-------------|----------------------------|----------------------|
| AB20        | [3]         | [3]                        | x                    |
| AEG20       | [1]         | [1]                        | x                    |
| BA12        | [4]         | [4]                        | x                    |
| BA14        | [4]         | [4]                        | x                    |
| BME15       | [6]         | [6]                        | x                    |
| D6          | [7]         | [7]                        | x                    |
| D8          | [7]         | [7]                        | x                    |
| D10         | [7]         | [7]                        | x                    |
| D12         | [7]         | [7]                        | x                    |
| FO7         | [15]        | [9]                        | x                    |
| FO8         | [15]        | [9]                        | x                    |
| FO9         | [15]        | [9]                        | x                    |
| FO10        | [14]        | [14]                       | x                    |
| FO11        | [14]        | [14]                       | x                    |
| LW5         | [13]        | [13]                       |                      |
| LW11        | [13]        | [13]                       |                      |
| M11         | [11]        | [11]                       | x                    |
| M15         | [11]        | [11]                       | x                    |
| M25         | [11]        | [11]                       | x                    |
| O7          | [15]        | [9]                        | x                    |
| O8          | [15]        | [9]                        | x                    |
| O9          | [5]         | [9]                        | x                    |
| O10         | [14]        | [14]                       | x                    |
| O12         | [5]         | [5]                        | x                    |
| P4          | [2]         | [2]                        |                      |
| P6          | [22]        |[22]                            | x                    |
| P12         |     [22]        | [22]                           | x                    |
| P15         | [2]         | [2]                        |                      |
| P20         | [10]        | [10]                       |                      |
| P30         | [10]        | [10]                       |                      |
| P62*        | [8]         | [8]                        |                      |
| S8          | [17]        | [17]                       | x                    |
| S8H         | [17]        | [17]                       | x                    |
| S9          | [17]        | [17]                       | x                    |
| S9H         | [17]        | [17]                       | x                    |
| SC30        | [12]        | [12]                       | x                    |
| SC35        | [12]        | [12]                       | x                    |
| TAM20       | [18]        | [18]                       | x                    |
| TAM30       | [18]        | [18]                       | x                    |
| TL5         | [16]        | [16]                       |                      |
| TL6         | [16]        | [16]                       |                      |
| TL7         | [16]        | [16]                       |                      |
| TL8         | [16]        | [16]                       |                      |
| TL12        | [16]        | [16]                       |                      |
| TL15        | [16]        | [16]                       |                      |
| TL20        | [16]        | [16]                       |                      |
| TL30        | [16]        | [16]                       |                      |
| VC10        | [19]        | [19]                       | x                    |
| WA7         | [21]        | [21]                       | x                    |
| WG6         | [20]        | [20]                       |                      |
| WG12        | [20]        | [20]                       |                      |

# References

[1] Aiello, G., Enea, M., Galante, G.: An integrated approach to the facilities
and material handling system design. International Journal of Production
Research 40(15), 4007–4017 (2002). DOI 10.1080/00207540210159572

[2] Amaral, A.R.: On the exact solution of a facility layout problem. European
Journal of Operational Research 173(2), 508–518 (2006). DOI 10.1016/j.
ejor.2004.12.021

[3] Armour, G.C., Buffa, E.S.: A heuristic algorithm and simulation approach
to relative location of facilities. Management Science 9(2), 294–309 (1963).
DOI 10.1287/mnsc.9.2.294

[4] Bazaraa, M.S.: Computerized layout design: A branch and bound approach.
A I I E Transactions 7(4), 432–438 (1975). DOI 10.1080/
05695557508975028

[5] Bozer, Y.A., Meller, R.D.: A reexamination of the distance-based facility
layout problem. IIE Transactions 29(7), 549–560 (1997). DOI 10.1080/
07408179708966365

[6] Bozer, Y.A., Meller, R.D., Erlebacher, S.J.: An improvement-type layout
algorithm for single and multiple-floor facilities. Management Science
40(7), 918–932 (1994). DOI 10.1287/mnsc.40.7.918

[7] Das, S.K.: A facility layout method for flexible manufacturing systems∗.
International Journal of Production Research 31(2), 279–297 (1993). DOI 10.1080/00207549308956725

[8] Dunker, T., Radons, G., Westkämper, E.: A coevolutionary algorithm for
a facility layout problem. International Journal of Production Research
41(15), 3479–3500 (2003). DOI 10.1080/0020754031000118125

[9] Gonçalves, J.F., Resende, M.G.: A biased random-key genetic algorithm
for the unequal area facility layout problem. European Journal of Operational
Research 246(1), 86–107 (2015). DOI 10.1016/j.ejor.2015.04.029

[10] Heragu, S.S., Kusiak, A.: Efficient models for the facility layout problem.
European Journal of Operational Research 53(1), 1–13 (1991). DOI 10.
1016/0377-2217(91)90088-D

[11] Komarudin, Wong, K.Y.: Applying ant system for solving unequal area facility
layout problems. European Journal of Operational Research 202(3),
730–746 (2010). DOI 10.1016/j.ejor.2009.06.016

[12] Liu, Q., Meller, R.D.: A sequence-pair representation and mip-modelbased
heuristic for the facility layout problem with rectangular departments.
IIE Transactions 39(4), 377–394 (2007). DOI 10.1080/
07408170600844108

[13] Love, R., Wong, J.: On solving a one-dimensional space allocation problem
with integer programming. INFOR: Information Systems and Operational
Research 14(2), 139–143 (1976). DOI 10.1080/03155986.1976.11731633

[14] Meller, R.D., Chen, W., Sherali, H.D.: Applying the sequence-pair representation
to optimal facility layout designs. Operations Research Letters
35(5), 651–659 (2007). DOI 10.1016/j.orl.2006.10.007

[15] Meller, R.D., Narayanan, V., Vance, P.H.: Optimal facility layout design.
Operations Research Letters 23(3-5), 117–127 (1998). DOI
10.1016/S0167-6377(98)00024-8

[16] Nugent, C.E., Vollmann, T.E., Ruml, J.: An experimental comparison of
techniques for the assignment of facilities to locations. Operations Research
16(1), 150–173 (1968). DOI 10.1287/opre.16.1.150

[17] Simmons, D.M.: One-dimensional space allocation: An ordering algorithm.
Operations Research 17(5), 812–826 (1969). DOI 10.1287/opre.17.5.812

[18] Tam, K.Y.R.: A simulated annealing algorithm for allocating space to
manufacturing cells. International Journal of Production Research 30(1),
63–87 (1992). DOI 10.1080/00207549208942878

[19] van Camp, D.J., Carter, M.W., Vannelli, A.: A nonlinear optimization
approach for solving facility layout problems. European Journal of Operational
Research 57(2), 174–189 (1992). DOI 10.1016/0377-2217(92)
90041-7

[20] Welgama, P.S., Gibson, P.R.: A construction algorithm for the machine
layout problem with fixed pick-up and drop-off points. International
Journal of Production Research 31(11), 2575–2589 (1993). DOI
10.1080/00207549308956884

[21] Wu, Y., Appleton, E.: The optimisation of block layout and aisle structure
by a genetic algorithm. Computers & Industrial Engineering 41(4), 371–
387 (2002). DOI 10.1016/S0360-8352(01)00063-8

[22] Yang, T., Peters, B.A.: Flexible machine layout design for dynamic and
uncertain production environments. European Journal of Operational Research
108(1), 49–64 (1998). DOI 10.1016/S0377-2217(97)00220-8

## Contribution

