Environmental Uncertainty → Flexibility
Uncertainty: Unpredictable variation in traffic load (car counts)
Quality Impact: Performance degrades as load increases (scalability issue)
Solution: Clustering + situation-specific optimization
Quality Attribute: Flexibility (3.8) - specifically Adaptability (3.8.1) and Scalability (3.8.2)

Model Uncertainty → Maintainability/Testability
Uncertainty: No analytical model relating parameters to performance
Quality Impact: Cannot predict impact of parameter changes (Analysability 3.7.3)
Solution: Black-box optimization (Bayesian) via runtime experimentation
Quality Attribute: Maintainability (3.7) - specifically Analysability (3.7.3) and Modifiability (3.7.4)

Parameter/Configuration Uncertainty → Performance Efficiency
Uncertainty: Unknown optimal values for 7-8 routing parameters
Quality Impact: Suboptimal Time Behaviour (3.2.1)
Solution: Optimization search in parameter space
Quality Attribute: Performance Efficiency (3.2) - Time Behaviour (3.2.1)

Ideas for our strategy:
Trade-off Uncertainty
Uncertainty: Unknown relationship between competing quality attributes
Quality Impact:
Time Behaviour (3.2.1) vs Resource Utilization (3.2.2): Lower trip overhead may require frequent re-routing hence higher computational cost
Paper 2 states: "multi-objective optimization problem with competing concerns"
Baseline Gap: Only optimizes trip overhead, ignores routing cost
Proposed Solution: Pareto optimization (NSGA-II or SMS-EGO)
Quality Attributes: Performance Efficiency (3.2) - balancing Time Behaviour (3.2.1) AND Resource Utilization (3.2.2)

User Satisfaction Uncertainty
Uncertainty: ?
Quality Impact: User Engagement (3.4.5) degrades → users may stop using system, more important than trip overhead
Baseline Gap: Complaints not optimized
Proposed Solution: add complaints as objective or as constraint
Quality Attribute: Interaction Capability (3.4) - User Engagement (3.4.5)

tick_duration
Maps to performance efficiency (3.2) - system throughput
- time per simulation tick (system-level metric)
- could indicate system strain under load
- implementation-specific, may not reflect managed system quality