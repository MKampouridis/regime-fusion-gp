# Java DC reference

`DCReference.java` reproduces the directional-change event detection of the group's earlier Java code: `generateEvents`, `detect2` and `adjust` from `misc/DCEventGenerator.java` in [adesolaadegboye/SymbolicRegression](https://github.com/adesolaadegboye/SymbolicRegression) (commit `af564a5`), copied unchanged apart from a reduced `Event` class.

`make_fixtures.py` generates five synthetic price series (`series/`), runs the Java code on them at five thresholds (0.5%, 1%, 2%, 5%, 10%) and saves the events to `events.csv`. The fixtures are committed, so the tests need no Java; regenerate them only if the series change (needs a JDK).

## Result (`tests/test_dc_java_crosscheck.py`)

- **Confirmation points match exactly** for all 6,798 events.
- **Start points (extremes) match for 3,825 events.** The other 2,973 differ in three edge cases of the Java code, each checked exactly by the test:
  1. An upturn with no lower low after the downturn's confirmation: the extreme is that confirmation point, but Java starts the upturn one bar later (1,254 events).
  2. A downturn with no higher high after the upturn's confirmation: Java reuses the start of the previous upturn, so the downturn appears to start before that upturn was confirmed (1,348 events). The cause is `adjust(..., indexOS, indexOS)` in the upturn branch, which passes the overshoot indices where the DC indices are expected.
  3. An upturn whose low comes exactly two bars after the downturn's confirmation: Java starts it one bar early (371 events).

Confirmation-based quantities therefore agree. Quantities built on start points, such as DC length, overshoot length and total move, differ in these cases.
