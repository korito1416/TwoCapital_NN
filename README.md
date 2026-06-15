# Two-stage jump model 

 

Run below code for solving the 6 HJB system:

sbatch -J TwoTechJumps parallel_handle.sbatch


Simulation Code is temporarily in the  /models/Simulation.ipynb file. 

<!-- 
## State Vairables for Different HJB equations

|                       | logK     | Z        |  Y       | logR     |   λ3     |   logξ   |
|-----------------------|----------|----------|----------|----------|----------|----------| 
|Post-Damage-Post-Tech  |    ✅    |   ✅     |   ✅     |          |    ✅    |    ✅    | 
|Post-Damage-Interm-Tech|    ✅    |   ✅     |   ✅     |   ✅     |    ✅    |    ✅    | 
|Post-Damage-Pre-Tech   |    ✅    |   ✅     |   ✅     |   ✅     |    ✅    |    ✅    | 
|Pre-Damage-Post-Tech   |    ✅    |   ✅     |   ✅     |          |          |    ✅    | 
|Pre-Damage-Interm-Tech |    ✅    |   ✅     |   ✅     |   ✅     |          |    ✅    | 
|Pre-Damage-Pre-Tech    |    ✅    |   ✅     |   ✅     |   ✅     |          |    ✅    | 

  -->