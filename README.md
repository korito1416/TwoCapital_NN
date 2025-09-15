66# Two-stage jump model 

 

Run below code for 
sbatch -J TwoTechJumps parallel_handle.sbatch



## State Vairables for Different HJB equations

|                       | logK     | Z        |  Y       | logR     |   λ3     |   logξ   |
|-----------------------|----------|----------|----------|----------|----------|----------| 
|Post-Damage-Post-Tech  |    ✅    |   ✅     |   ✅     |          |    ✅    |    ✅    | 
|Post-Damage-Interm-Tech|    ✅    |   ✅     |   ✅     |   ✅     |    ✅    |    ✅    | 
|Post-Damage-Pre-Tech   |    ✅    |   ✅     |   ✅     |   ✅     |    ✅    |    ✅    | 
|Pre-Damage-Post-Tech   |    ✅    |   ✅     |   ✅     |          |          |    ✅    | 
|Pre-Damage-Interm-Tech |    ✅    |   ✅     |   ✅     |   ✅     |          |    ✅    | 
|Pre-Damage-Pre-Tech    |    ✅    |   ✅     |   ✅     |   ✅     |          |    ✅    | 

 