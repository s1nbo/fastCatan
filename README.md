# fastCatan
Catan Simulator for Reinforcement Learning in Stochastic Multi-Agent Environments: Mastering Settlers of Catan



 Remaining                                                                                                                                                                                                          
                                                         
  ┌─────────────────────────────────────────────────────────┬───────────────────────────────────────────────────┐
  │                          Item                           │                       Notes                       │
  ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
  │ Player-to-player trading (M2)                           │ Sub-phase compositional encoding per PLAN.md L148 │
  ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
  │ Action ID translation for byte-exact diff vs catanatron │ M2; substantial mapping work                      │                                                                                                    
  ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤                                                                                                    
  │ write_obs (obs.cpp)                                     │ Encode GameState → flat tensor for RL agent input │                                                                                                    
  ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤                                                                                                    
  │ Gymnasium wrapper                                       │ Wraps env for RL libraries                        │
  ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤                                                                                                    
  │ Incremental mask + longest-road (M3)                    │ Perf optimization, not correctness                │
  └─────────────────────────────────────────────────────────┴───────────────────────────────────────────────────┘    