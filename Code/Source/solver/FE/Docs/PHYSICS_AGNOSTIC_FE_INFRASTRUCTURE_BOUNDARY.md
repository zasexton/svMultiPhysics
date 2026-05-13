# Physics-Agnostic FE Infrastructure Boundary

This note defines the ownership boundary for the reusable FE infrastructure added for complex coupled simulations. The FE layer provides metadata, storage, generic checks, reductions, accumulators, and geometry query services. Physics modules provide governing equations, constitutive laws, closure correlations, source terms, and domain-specific output names.

## What FE Owns

FE infrastructure may define generic concepts that are meaningful across many applications:

- state groups and component-set metadata
- admissibility descriptors, invariant-domain summaries, and bounded update policies
- equal-and-opposite paired exchange bookkeeping
- threshold, histogram, percentile, and min/max reductions
- accumulated exposure of an arbitrary scalar quantity
- boundary-distance queries and nearest-boundary metadata
- coordinate-frame descriptors, region-to-frame bindings, and sliding-interface transfer metadata

These APIs should remain opt-in. Default behavior for existing physics should be unchanged unless a physics module explicitly registers one of these descriptors or services.

## What Physics Owns

Physics modules decide the meaning of a field, source, closure, threshold, or output. FE should not know whether a scalar represents concentration, temperature, density, volume fraction, stress, or a biological state. FE should only know the shape, region, bounds, storage scope, and generic operation requested by the physics module.

Examples:

- A bounded concentration-like scalar can use `StateAdmissibilityDescriptor` with an interval check, but the concentration equation, units, source law, and threshold choice belong to Physics.
- A generic two-field exchange can use `PairedExchangeDescriptor` to report equal-and-opposite balance metadata, but the exchange rate law and coefficients belong to Physics.
- The threshold volume of an arbitrary scalar can use `ThresholdReductionDefinition`, but the critical value and the scalar's interpretation belong to Physics.
- The accumulated exposure of an arbitrary field can use `ExposureAccumulator`, but the hazardous quantity and exposure threshold belong to Physics.

## Do Not Put In FE

The following are examples of model-specific content that should remain outside the FE library:

- oxygen transfer laws
- bubble breakup or coalescence laws
- gas-liquid drag laws
- RANS closures
- MRF source terms
- cell damage models
- bioreactor-specific metrics

FE can provide the reusable hooks these models consume: fields, state groups, admissibility checks, reductions, exposure tracking, boundary-distance queries, frame bindings, and interface-transfer metadata. The model formulas and their validation remain the responsibility of the physics module.
