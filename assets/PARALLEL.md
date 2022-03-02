## Performance effected by parallel training 

1. Select a few random agents to update the current agent
- small effect, especially with good algorithm, there is in-significant between sequential or parallel training mode

``` 
A. Evolutionary
    1. GA: Wheel selection
    2. FPA: Random 2 agents in population
    3. DE 
        BaseDE: select random depend on the selected strategy
        JADE: select based permutation sorted and randint
        SADE: select random agents from permutation
        SHADE, L_SHADE: select random agents from permutation
        SAP-DE: select random agents
    4. CRO: 
        + BaseCRO: Mixed methods
        + OCRO: Mixed methods
     
B. Bio
    1. BBO (BaseBBO, OriginalBBO): Select roulette wheel
    2. EOA: Select 2 random agents from population
    3. SBO: Select agent based on roulette wheel
    4. SMA: Select 2 random agents from population

E. System
    1. GCO: Select 2 random agents from population
    2. WCA: Depend on list_pop_best
    
F. Human
    1. BRO: Select 1 solution based on minimum distance in population
    2. BSO: Select solutions from different group
    3. SARO: Random 2 agents from population
    4. TLO: 
        + BaseTLO, OriginalTLO, ITLO
        + Based on different mean, random, different teachers
    5. QSA: Mixed of few methods to choice agents 
        + BaseQSA
        + OppoQSA
        + LevyQSA
        + ImprovedQSA
        + OriginalQSA

G. Physics
    1. ArchOA: Random 1 agent
    2. MVO (BaseMVO, OriginalMVO)
    3. SA 
    4. WDO 
    5. TWO (BaseTWO, OppoTWO, LevyTWO, ImprovedTWO)
    6. NRO: Mixed method

H. Swarm
    1. ACOR: Select random solution
    2. BES: Mixed methods 
    3. BSA: Mixed methods
    4. CSA
    5. NMRA: 
        + BaseNMRA: Random 2 agents
        + ImprovedNMRA: Mixed methods
    6. PSO:
        + BasePSO
        + PPSO
        + C-PSO
        + CL-PSO
        + HPSO-TVAC
    7. SRSR: Mixed methods (too complex)

I. Probability
    1. CEM: Mean of population
    
    
```

2. Select a previous or next agents to update the current agent
- big effect, some time not much but in general will be a huge effect to the results

```
A. Evolutionary
    1. MA: Select i-1 and i+1 as ancient
    
B. Bio
    1. VCS: 
        BaseVCS: Select random + mean(population)
        OriginalVCS: Selection random + mean(population)

F. Human
    1. LCO: (OriginalLCO, BaseLCO, ImprovedLCO)
        + Select previous solution to update current agent
    2. GSKA: Select previous and next agents to update current agent
    3. ICA: Select different colonies and empires to update
    4. LCO: Select previous and next agents to update current agent
    5. SSDO: Mean of three best solutions
    6. CHIO: 
        + OriginalCHIO: too weak
        + BaseCHIO: mixed methods

G. Physics
    1. EFO: 
        + OriginalEFO: Change only 1 solution in population --> weak
        + BaseEFO: My version based on swarm-inspired    
    2. HGSO: 

H. Swarm
    1. BA: OriginalBA, BasicBA (Both papers belong to this category)
    2. CSO: Mixed methods
    3. DO: Mixed methods
    4. EHO: Mean group 
    5. FA: 
    6. FireflyA: Mixed methods
    7. HHO: Mixed methods
    8. MFO: So weak (OriginalMFO)
    9. PFA: Mean of all previous population
    10. SSO: Select previous agent
  
    
```

3. Using local best or global best to update current agent
- no effect at all

```
A. Evolutionary
    1. ES (BaseES, LevyES) 
    2. EP (BaseEP, LevyEP)

B. Bio
    1. IWO (OriginalIWO)
    2. WHO (BaseWHO): strong due to the new strategy 

C. Math
    1. AOA 
    2. HC:
        OriginalHC: the version is built based on neighborhood size
        BasicHC: my version is built based on swarm-based idea
    3. SCA

D. Music
    1. HS (OrigianlHS and BaseHS)

E. System
    1. AEO (BaseAEO, ImprovedAEO, ModifiedAEO, AdaptiveAEO, EnhancedAEO)
        + Select random agents in population but strong algorithm in the end.

F. Human
    1. CA
    
G. Physics
    1. ASO

H. Swarm
    1. ABC: Neighborhood search
    2. ALO (BaseALO, OriginalALO)
    3. AO
    4. BA: BaseBA - my version 
    5. BeesA: BaseBeesA and ProbBeesA version
    6. FOA:
        + OriginalFOA: Week, 
        + BaseFOA: Week too
        + WhaleFOA: Better than these two versions
    7. GOA 
    8. GWO 
    9. HGS 
    10. JA (OriginalJA, BaseJA, LevyJA)
    11. MFO (BaseMFO)
    12. MRFO: Mixed method but strong algorithm
    13. MSA: Levy-flight
    14. SFO: Two population (sailfish and sardine)
    15. SHO: Local search --> Strong
    16. SLO: BaseSLO, ModifiedSLO, ISLO 
    17. SSA: BaseSSA, OriginalSSA 
    18. WOA: BaseWOA, HI_WOA
    19. SSpiderA
    20. SSpiderO: Pretty good
    21. COA 
    22. BFO: Local search (both OriginalBFO, ABFO)
    
```

